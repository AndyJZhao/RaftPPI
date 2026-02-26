import os
import os.path
import pickle
import random
import logging

import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from src.utils.log import timer

logger = logging.getLogger(__name__)


def load_hf_dataset_from_local_if_possible(remote_hf_path, local_path, token=None, **kwargs):
    if local_path and os.path.exists(local_path):
        return load_from_disk(local_path, **kwargs)

    ds = load_dataset(remote_hf_path, token=token, **kwargs)

    # Mirror downloaded datasets to project-local storage so subsequent loads
    # (including later stages in the same run) can use load_from_disk directly.
    if local_path:
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            ds.save_to_disk(local_path)
        except Exception as exc:
            # Fallback to HF cache only when local mirroring fails.
            logger.warning(f"Failed to mirror dataset to local path {local_path}: {exc}")

    return ds

def pickle_save(var, file_name: str):
    directory = os.path.dirname(file_name)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(file_name, "wb") as file:
        pickle.dump(var, file)


def pickle_load(file_name: str):
    with open(file_name, "rb") as file:
        return pickle.load(file)

###########################################################
# Data Collator
###########################################################
class PPITripletCollator:
    def __init__(self, max_length=1024, tokenizer=None):
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # batch is a list of ([tokenized_receptor_dict, tokenized_ligand_dict], label)
        # tokenized_receptor_dict and tokenized_ligand_dict are like {'input_ids': [...], 'attention_mask': [...]}

        # Extract pre-tokenized inputs
        r_tokenized_inputs = [triplet[0][0] for triplet in batch]
        l_tokenized_inputs = [triplet[0][1] for triplet in batch]
        # Keep labels on CPU
        prot_labels = torch.tensor([triplet[1] for triplet in batch], dtype=torch.float32)

        # Pad the tokenized sequences.
        # The PPIDataset already truncated raw sequences to max_length before tokenization.
        # Here, we pad to self.max_length. t0runcation=True is a safeguard.
        r_tokens = self.tokenizer.pad(
            r_tokenized_inputs,
            padding=True,
            return_tensors="pt",
        )
        l_tokens = self.tokenizer.pad(
            l_tokenized_inputs,
            padding=True,
            return_tensors="pt",
        )

        return {
            'r_tokens': r_tokens,
            'l_tokens': l_tokens,
            'labels': prot_labels  # for protein-level loss computation
        }


class PPIDataset:
    def __init__(self, dataset, cache_file: str, tokenizer, max_length: int,
                 num_negatives=1, max_data_samples=None,
                 l_key='prot_1', r_key='prot_2', *args, **kwargs):
        self.dataset = dataset  # HF dataset object
        self.num_negatives = num_negatives
        self.tokenizer = tokenizer
        self.max_seq_length = max_length
        self.l_key, self.r_key = l_key, r_key
        self.cache_file = cache_file

        if not os.path.exists(cache_file):
            self.logger = kwargs.get('logger', None)  # Optional logger for progress
            if self.logger:
                self.logger.info(f"Cache file {cache_file} not found. Building cache...")

            # prot_str_to_id and prot_id_to_str can be removed if not used elsewhere,
            # as we primarily care about tokenized pairs now.
            # For now, keep them if other parts of your system rely on raw sequence identity.
            self.prot_str_to_id, self.prot_id_to_str = {}, {}
            for ppi_pair in tqdm(self.dataset,
                                 desc=f'Building unique raw sequence map for {os.path.basename(self.cache_file)}'):
                for seq in (ppi_pair[self.r_key], ppi_pair[self.l_key]):
                    if seq not in self.prot_str_to_id:
                        self.prot_str_to_id[seq] = len(self.prot_str_to_id)
                        self.prot_id_to_str[self.prot_str_to_id[seq]] = seq

            self.protein_pairs = self._build_prot_pos_neg_pairs_tokenized(max_data_samples)
            pickle_save((self.prot_str_to_id, self.prot_id_to_str, self.protein_pairs), self.cache_file)
            if self.logger: self.logger.info(f"Cache built and saved to {self.cache_file}")
        else:
            self.prot_str_to_id, self.prot_id_to_str, self.protein_pairs = pickle_load(self.cache_file)
            if kwargs.get('logger', None):
                kwargs['logger'].info(f"Loaded tokenized PPI pairs from cache: {self.cache_file}")

    def _tokenize_sequence(self, sequence: str):
        # Tokenize but do not pad here. Padding will be done per batch by the collator.
        # Return as dict of lists (input_ids, attention_mask)
        tokenized_output = self.tokenizer(
            sequence[:self.max_seq_length],
            truncation=True,
            padding=False,  # Important: No padding here
            return_attention_mask=True
        )
        return {
            'input_ids': tokenized_output['input_ids'],
            'attention_mask': tokenized_output['attention_mask']
        }

    def _build_prot_pos_neg_pairs_tokenized(self, max_data_samples):
        desc_prefix = f"Tokenizing and building pairs for {os.path.basename(self.cache_file)} - "
        positive_pairs_tokenized = []
        # Keep track of original raw positive pairs for negative sampling if needed for logic
        _raw_positive_pairs_for_neg_sampling = []

        for sample in tqdm(self.dataset, desc=desc_prefix + 'positive'):
            prot1_raw = sample[self.r_key]
            prot2_raw = sample[self.l_key]

            tokenized_prot1 = self._tokenize_sequence(prot1_raw)
            tokenized_prot2 = self._tokenize_sequence(prot2_raw)

            positive_pairs_tokenized.append((tokenized_prot1, tokenized_prot2))
            _raw_positive_pairs_for_neg_sampling.append((prot1_raw, prot2_raw))  # Store raw for neg sampling logic
            if max_data_samples is not None and len(positive_pairs_tokenized) >= max_data_samples:
                break

        negative_pairs_tokenized = []
        n_pos = len(_raw_positive_pairs_for_neg_sampling)

        for i in tqdm(range(n_pos), desc=desc_prefix + 'negative'):
            for _ in range(self.num_negatives):
                j = i
                while j == i:  # Ensure different partner for negative sample
                    j = random.randint(0, n_pos - 1)

                # Negative pair: prot1 from positive pair i, prot2 from positive pair j
                neg_prot1_raw = _raw_positive_pairs_for_neg_sampling[i][0]
                neg_prot2_raw = _raw_positive_pairs_for_neg_sampling[j][1]

                tokenized_neg_prot1 = self._tokenize_sequence(neg_prot1_raw)
                tokenized_neg_prot2 = self._tokenize_sequence(neg_prot2_raw)
                negative_pairs_tokenized.append((tokenized_neg_prot1, tokenized_neg_prot2))

        return positive_pairs_tokenized, negative_pairs_tokenized

    def to_ppi_triplets(self):
        triplets = []
        pos_pairs_tok, neg_pairs_tok = self.protein_pairs  # These are now tokenized
        for tokenized_protA, tokenized_protB in pos_pairs_tok:
            triplets.append(([tokenized_protA, tokenized_protB], 1))
        for tokenized_protA, tokenized_protB in neg_pairs_tok:
            triplets.append(([tokenized_protA, tokenized_protB], 0))
        random.shuffle(triplets)
        return triplets


class DscriptDataset(PPIDataset):
    def __init__(self, dataset, cache_file: str, tokenizer, max_length: int,
                 max_data_samples=None, *args, **kwargs):
        # Pass tokenizer and max_length to parent
        super().__init__(dataset, cache_file, tokenizer, max_length,
                         max_data_samples=max_data_samples, *args, **kwargs)

    def _build_prot_pos_neg_pairs_tokenized(self, max_data_samples):
        desc_prefix = f"Tokenizing for {os.path.basename(self.cache_file)} - "
        positive_pairs_tokenized, negative_pairs_tokenized = [], []

        # Process positive pairs
        pos_count = 0
        for sample in tqdm(self.dataset, desc=desc_prefix + 'positives'):
            if sample['label'] == 1:
                tokenized_prot1 = self._tokenize_sequence(sample[self.r_key][:self.max_seq_length])
                tokenized_prot2 = self._tokenize_sequence(sample[self.l_key][:self.max_seq_length])
                positive_pairs_tokenized.append((tokenized_prot1, tokenized_prot2))
                pos_count += 1
                if max_data_samples is not None and pos_count >= max_data_samples:
                    break  # Stop collecting positive pairs if limit reached

        # Process negative pairs (up to max_data_samples if specified, or all available negatives)
        neg_count = 0
        # Iterate again or use a pre-filtered negative list if dataset is large and label access is slow.
        # For now, simple iteration.
        for sample in tqdm(self.dataset, desc=desc_prefix + 'negatives'):
            if sample['label'] == 0:
                tokenized_prot1 = self._tokenize_sequence(sample[self.r_key])
                tokenized_prot2 = self._tokenize_sequence(sample[self.l_key])
                negative_pairs_tokenized.append((tokenized_prot1, tokenized_prot2))
                neg_count += 1
                # If max_data_samples applies to total samples, this logic might need adjustment.
                # Assuming max_data_samples here applies per class (pos/neg) if they are collected separately.
                # If it means total samples, then one needs to be more careful.
                # For DScript it usually means count for positive AND count for negative.
                if max_data_samples is not None and neg_count >= max_data_samples:
                    break

        return positive_pairs_tokenized, negative_pairs_tokenized



@timer()
def load_single_dscript_dataset(cfg, logger, tokenizer):
    dataset_dict = {}
    dataset_cfg = cfg.dataset
    ds_dict = load_hf_dataset_from_local_if_possible(
        dataset_cfg.hf_path,
        dataset_cfg.local_path,
        token=cfg.hf_access_token
    )

    for split, dataset in ds_dict.items():
        data_alias = f'{split}_{cfg.data}'
        cache_root = cfg.get("dirs", {}).get("data_cache", "data_cache")
        cache_file = f'{cache_root}/{data_alias}_max_data_samples={cfg.get("max_data_samples")}' + \
                     f'_data_tokenized_cache.pkl'
        dataset_dict[split] = DscriptDataset(dataset, cache_file, tokenizer, max_length=cfg.max_length,
                                             max_data_samples=cfg.get('max_data_samples'))

    return dataset_dict
