from typing import Dict

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from scipy.linalg import hadamard
from transformers import AutoTokenizer, AutoModel


class SORFLayer(nn.Module):
    def __init__(self, input_dim, rff_dim=2048, sigma=1.0, seed=0, float_dtype=torch.bfloat16):
        """
        Args:
            input_dim: Dimension of input protein embeddings
            rff_dim: Desired RFF output dimension (final output will be 2 * rff_dim)
            sigma: Bandwidth of RBF kernel
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.input_dim = input_dim
        self.rff_dim = rff_dim
        self.sigma = sigma
        self.seed = seed

        self.sorf_base_dim = self.get_next_power_of_2(input_dim)
        self.num_blocks = math.ceil(rff_dim / self.sorf_base_dim)
        self.sorf_matrices = nn.ParameterList()

        for i in range(self.num_blocks):
            mat = self._create_sorf_matrix(self.sorf_base_dim, sigma, seed + i, float_dtype)
            self.sorf_matrices.append(nn.Parameter(mat, requires_grad=False))

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, input_dim)
        returns: Tensor of shape (batch_size, 2 * rff_dim)
        """
        batch_size = x.size(0)
        device = x.device

        if x.size(1) < self.sorf_base_dim:
            pad_width = self.sorf_base_dim - x.size(1)
            x = torch.cat([x, torch.zeros(batch_size, pad_width, device=device, dtype=x.dtype)], dim=1)

        cos_blocks = []
        sin_blocks = []
        for i in range(self.num_blocks):
            sorf_mat = self.sorf_matrices[i].to(device)
            z = x @ sorf_mat.T  # (batch, d)
            cos_z = torch.cos(z)
            sin_z = torch.sin(z)
            cos_blocks.append(cos_z)
            sin_blocks.append(sin_z)

        cos_cat = torch.cat(cos_blocks, dim=1)[:, :self.rff_dim]
        sin_cat = torch.cat(sin_blocks, dim=1)[:, :self.rff_dim]

        output = torch.cat([cos_cat, sin_cat], dim=1) / np.sqrt(self.rff_dim)
        return output

    @staticmethod
    def get_next_power_of_2(n):
        return 1 << (n - 1).bit_length()

    @staticmethod
    def _create_sorf_matrix(d, sigma=1.0, seed=None, float_dtype=torch.bfloat16):
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
        else:
            g = None

        if d & (d - 1) != 0:
            raise ValueError(f"Dimension {d} must be a power of 2")

        # H is still deterministic; ensure it's on consistent device later
        H = torch.tensor(hadamard(d), dtype=float_dtype) / np.sqrt(d)

        D1 = torch.diag(2 * torch.randint(0, 2, (d,), generator=g).to(float_dtype) - 1)
        D2 = torch.diag(2 * torch.randint(0, 2, (d,), generator=g).to(float_dtype) - 1)
        D3 = torch.diag(2 * torch.randint(0, 2, (d,), generator=g).to(float_dtype) - 1)

        sorf_mat = (np.sqrt(d) / sigma) * H @ D1 @ H @ D2 @ H @ D3
        return sorf_mat


class RaftModel(nn.Module):
    def __init__(self, cfg: DictConfig, logger, float_dtype):
        super().__init__()
        self.cfg = cfg
        self.logger = logger

        hf_ckpt = getattr(cfg, 'local_hf_ckpt', cfg.hf_ckpt) if cfg.get('offline_mode', False) else cfg.hf_ckpt

        self.esm = AutoModel.from_pretrained(hf_ckpt, trust_remote_code=True)
        logger.info("Using full fine-tuning")

        self.tokenizer = AutoTokenizer.from_pretrained(hf_ckpt, trust_remote_code=True)

        self.hidden_size = hidden_size = self.esm.config.hidden_size
        self.prot_readout = cfg.prot_readout
        self.attn_rank = max(1, int(self.cfg.get('attn_rank', 1)))
        self.prot_emb_norm = bool(self.cfg.get('prot_emb_norm', False))
        self.res_emb_norm = bool(self.cfg.get('res_emb_norm', True))
        if self.prot_readout not in {'attn', 'mlp_attn'} and self.attn_rank != 1:
            logger.warning(
                f"attn_rank={self.attn_rank} is only supported for attention-based readouts. "
                f"Resetting to 1 for prot_readout={self.prot_readout}."
            )
            self.attn_rank = 1
        self.kernel_sigma = self.cfg.get('sigma', 1.0)
        self.loss_type = self.cfg.get('loss_type', 'BCE')
        self.use_sorf = self.cfg.get('use_sorf', True)
        if self.use_sorf:
            self.sorf_layer = SORFLayer(hidden_size, cfg.rff_dim, self.kernel_sigma, cfg.seed, float_dtype)

        if self.prot_readout == 'attn':
            self.query_proj = nn.Linear(hidden_size, hidden_size * self.attn_rank)
            self.key_proj = nn.Linear(hidden_size, hidden_size)
        elif self.prot_readout == 'mlp_attn':
            self.mlp_attn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.attn_rank)
            )

    def _get_residue_emb_and_weights(self, tokens):
        """
        Extracts L2-normalized residue embeddings and aggregation weights.
        Both are returned for the FULL token sequence (including BOS/EOS).
        Weights are normalized to sum to 1 over the token sequence length (considering padding via attention_mask).

        Returns:
            residue_embs_full (torch.Tensor): (B, L_full, D)
            weights_full (torch.Tensor): (B, attn_rank, L_full)
        """

        outputs = self.esm(**tokens)
        hidden = outputs.last_hidden_state  # (B, L_full, D)
        attention_mask = tokens['attention_mask'].to(device=hidden.device, dtype=hidden.dtype)

        masked_hidden = hidden * attention_mask.unsqueeze(-1)
        if self.res_emb_norm:
            norm = masked_hidden.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            residue_embs = torch.where(
                attention_mask.unsqueeze(-1).bool(),
                masked_hidden / norm,
                torch.zeros_like(masked_hidden)
            )  # (B, L_full, D)
        else:
            residue_embs = torch.where(
                attention_mask.unsqueeze(-1).bool(),
                masked_hidden,
                torch.zeros_like(masked_hidden)
            )  # (B, L_full, D)

        attn_mask_expanded = attention_mask.unsqueeze(1)  # (B, 1, L_full)
        batch_size, seq_len, _ = residue_embs.shape

        if self.prot_readout == 'attn':
            keys = self.key_proj(residue_embs)  # (B, L_full, D)
            cls_feats = residue_embs[:, 0, :]  # (B, D)
            queries = self.query_proj(cls_feats)  # (B, D * attn_rank)
            queries = queries.view(batch_size, self.attn_rank, -1)
            scale = queries.size(-1) ** 0.5
            attn_scores = torch.einsum('brd,bld->brl', queries / scale, keys)  # (B, r, L_full)
            attn_scores = attn_scores.masked_fill(attn_mask_expanded == 0, float('-inf'))
            weights_full = attn_scores.softmax(dim=-1)  # (B, r, L_full)

        elif self.prot_readout == 'mlp_attn':
            mlp_scores = self.mlp_attn(residue_embs)  # (B, L_full, r)
            mlp_scores = mlp_scores.permute(0, 2, 1)  # (B, r, L_full)
            mlp_scores = mlp_scores.masked_fill(attn_mask_expanded == 0, float('-inf'))
            weights_full = mlp_scores.softmax(dim=-1)  # (B, r, L_full)

        elif self.prot_readout == 'cls':
            weights_full = torch.zeros(batch_size, self.attn_rank, seq_len, device=hidden.device, dtype=hidden.dtype)
            weights_full[:, 0, 0] = 1.0

        elif self.prot_readout == 'avg':
            counts = attention_mask.sum(dim=1, keepdim=True).clamp_min(1e-9)
            base_weights = (attention_mask / counts).unsqueeze(1)  # (B, 1, L_full)
            if self.attn_rank == 1:
                weights_full = base_weights
            else:
                weights_full = base_weights.repeat(1, self.attn_rank, 1)
        else:
            raise ValueError(f"Unknown prot_readout: {self.prot_readout}")

        weights = weights_full * attn_mask_expanded
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return residue_embs, weights

    def forward(self, r_tokens: Dict[str, torch.Tensor], l_tokens: Dict[str, torch.Tensor], labels: torch.Tensor,
                mode='training'):
        prot_scores = self._compute_scores(r_tokens, l_tokens)
        if mode == 'training':
            prot_loss = self._compute_loss(prot_scores, labels)
        elif mode == 'inference':
            prot_loss = None
        else:
            raise ValueError(f'Unknown {mode=}')
        return prot_scores, prot_loss

    def get_protein_embedding(self, tokens):
        # Use this for PPI screening over proteome
        res_z, r_weights = self._get_residue_emb_and_weights(tokens)  # (B, L, D), (B, r, L)
        prot_emb = torch.einsum('brl,bld->brd', r_weights, res_z)  # (B, r, D)

        batch_size = prot_emb.size(0)
        prot_emb = prot_emb.reshape(batch_size * self.attn_rank, self.hidden_size)

        if self.use_sorf:
            # Gaussian Embedding to approximate RKHS kernel
            prot_emb = self.sorf_layer(prot_emb)

        prot_emb = prot_emb.view(batch_size, self.attn_rank, -1)
        prot_emb = prot_emb.reshape(batch_size, -1)
        if self.prot_emb_norm:
            prot_emb = F.normalize(prot_emb, p=2, dim=-1)
        return prot_emb

    def _compute_scores(self, r_tokens, l_tokens):
        r_prot_emb = self.get_protein_embedding(r_tokens)
        l_prot_emb = self.get_protein_embedding(l_tokens)

        prot_scores = torch.sum(r_prot_emb * l_prot_emb, dim=-1)
        return prot_scores

    def _compute_loss(self, prot_scores, labels):
        target_device = labels.device
        if self.loss_type == "BCE":
            loss_fn = nn.BCEWithLogitsLoss(reduction='mean')  # Ensure reduction is 'mean'
            prot_loss = loss_fn(prot_scores, labels)
        elif self.loss_type == 'Ranking':
            pos_mask = labels == 1
            neg_mask = labels == 0

            pos_logits = prot_scores[pos_mask]
            neg_logits = prot_scores[neg_mask]

            loss_pos = torch.tensor(0.0, device=target_device)
            if pos_logits.numel() > 0:
                loss_pos = -F.logsigmoid(pos_logits).mean()  # Ensure mean reduction

            loss_neg = torch.tensor(0.0, device=target_device)
            if neg_logits.numel() > 0:
                adv_temp = self.cfg.get('adv_temp', 4.0)
                weights_n = F.softmax(neg_logits * adv_temp, dim=0).detach()
                loss_neg = -(weights_n * F.logsigmoid(-neg_logits)).sum()  # weights_n already sums to 1

            if pos_logits.numel() > 0 or neg_logits.numel() > 0:
                prot_loss = (loss_pos + loss_neg) / 2.0
            else:
                prot_loss = torch.tensor(0.0, device=target_device, requires_grad=True)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}.")

        return prot_loss
