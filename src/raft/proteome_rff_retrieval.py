import os
import pickle
import math
import time
from typing import Dict, List, Set, Optional, Tuple

import chromadb
import numpy as np
import rootutils
import torch
from chromadb.config import Settings
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoTokenizer

from src.raft.data import load_hf_dataset_from_local_if_possible
from src.utils.experiment import get_device

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)


def evaluate_ranking(y_true, y_pred):
    # Compute AUC on continuous (float) predictions.
    auc_test = float(roc_auc_score(y_true, y_pred))
    scores = {f'auc': round(auc_test, 4)}
    return scores

# ============================================================================
# Utility Functions
# ============================================================================

def pickle_save(data, file_path: str):
    """Save data to pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def pickle_load(file_path: str):
    """Load data from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def read_proteome_fasta(fasta_path: str, logger) -> dict:
    """Read proteome from FASTA file and return uniprot_id -> sequence mapping."""
    logger.info(f"Reading proteome from FASTA: {fasta_path}")
    proteome_data = {}
    current_uniprot_id = ''
    current_seq_parts = []
    
    try:
        with open(fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('>'):
                    # Save previous sequence
                    if current_uniprot_id and current_seq_parts:
                        proteome_data[current_uniprot_id] = "".join(current_seq_parts)
                    
                    # Parse new header - assumes ID is the first word after '>'
                    header_parts = line[1:].split()
                    if header_parts:
                        current_uniprot_id = header_parts[0].split('|')[1] if '|' in header_parts[0] else header_parts[0]
                    else:
                        current_uniprot_id = ""
                    current_seq_parts = []
                else:
                    if current_uniprot_id:  # Only append if we are under an ID
                        current_seq_parts.append(line)
                        
            # Add the last sequence
            if current_uniprot_id and current_seq_parts:
                proteome_data[current_uniprot_id] = "".join(current_seq_parts)
                
    except FileNotFoundError:
        logger.error(f"Proteome FASTA file not found: {fasta_path}")
        raise
        
    logger.info(f"Loaded {len(proteome_data)} UniProt ID-sequence pairs from {fasta_path}.")
    return proteome_data


# ============================================================================
# Recall Calculation Functions
# ============================================================================

def calculate_recall_at_top_x_percent(retrieved_ids: List[int], true_partner_ids: Set[int], 
                                     percentage_values_list: List[float], total_candidates: int, 
                                     logger) -> Dict[str, Tuple[int, int]]:
    """Calculate recall@top-X% for embedding-based retrieval (HNSW/ChromaDB)."""
    recall_results = {}
    if not true_partner_ids:
        for p_val in percentage_values_list:
            recall_results[f"recall@{p_val}"] = (0, 0)  # (hits, total)
        return recall_results

    true_partner_set = set(true_partner_ids)
    num_true_partners = len(true_partner_set)
    if num_true_partners == 0:
        for p_val in percentage_values_list:
            recall_results[f"recall@{p_val}"] = (0, 0)
        return recall_results

    for p_val in percentage_values_list:
        if not (0 < p_val <= 100):
            logger.warning(f"Invalid percentage {p_val} for Recall@TopX%. Skipping.")
            continue

        k = math.ceil((p_val / 100.0) * total_candidates)
        top_k_predictions = retrieved_ids[:k]
        top_k_set = set(top_k_predictions)

        # Count how many true partners are in the top-k predictions
        hits = sum(1 for partner in true_partner_set if partner in top_k_set)

        # Return hits and total for later aggregation
        recall_results[f"recall@{p_val}"] = (hits, num_true_partners)

    return recall_results


# ============================================================================
# Dataset Class
# ============================================================================

class ProteomeRankingDataset:
    """Unified dataset class for protein-protein interaction ranking evaluation."""
    
    def __init__(self, cfg: DictConfig, logger, tokenizer: AutoTokenizer, max_length: int,
                 proteome_fasta_paths: Dict[str, str], query_hf_dataset_cfg: Optional[DictConfig] = None):
        self.cfg = cfg
        self.logger = logger
        self.query_hf_dataset_cfg = query_hf_dataset_cfg or cfg.dataset
        self._tokenizer_for_processing = tokenizer
        self._max_length_for_processing = max_length
        self.proteome_fasta_paths = proteome_fasta_paths

        # Determine organism based on dataset name
        query_dataset_name = self.cfg.data
        if query_dataset_name.lower() in ['du', 'guo']:
            self.organism = 'yeast'
        elif query_dataset_name.lower() in ['huang', 'dscript', 'pan', 'richoux', 'gold', 'human']:
            self.organism = 'human'
        else:
            self.logger.warning(
                f"Could not determine organism for query dataset '{query_dataset_name}'. Defaulting to human.")
            self.organism = cfg.get("default_proteome_organism", "human")

        # Setup caching
        proteome_name_for_cache = f"proteome_{self.organism}_uniprot"
        cache_dir_name = (f"ranking_ppi_{proteome_name_for_cache}_queries_{query_dataset_name}_num_ranking_eval_"
                          f"{cfg.get('num_ranking_eval', 'all')}")
        cache_dir = os.path.join(root, "data_cache", cache_dir_name)
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, f"{query_dataset_name}_tokenized_max_len_{max_length}_ids.pkl")

        # Load or build dataset
        if os.path.exists(self.cache_file) and not cfg.get("ignore_cache", False):
            self._load_from_cache()
        else:
            self._load_and_process_dataset()
            self._save_to_cache()

        # Set convenience attributes
        self.prot_id_to_str = self.internal_id_to_proteome_seq
        self.prot_str_to_id = {seq: i for i, seq in self.internal_id_to_proteome_seq.items()}

    def _load_from_cache(self):
        """Load dataset from cache."""
        self.logger.info(f"Loading ProteomeRankingDataset from cache: {self.cache_file}")
        cached_data = pickle_load(self.cache_file)
        
        self.uniprot_to_internal_id = cached_data['uniprot_to_internal_id']
        self.internal_id_to_proteome_seq = cached_data['internal_id_to_proteome_seq']
        self.all_unique_input_ids = cached_data['all_unique_input_ids']
        self.all_unique_attention_masks = cached_data['all_unique_attention_masks']
        self.query_master_indices = cached_data['query_master_indices']
        self.ground_truth_map_query_id_to_partner_ids = cached_data['ground_truth_map_query_id_to_partner_ids']
        self.total_possible_query_uniprots = cached_data.get('total_possible_query_uniprots',
                                                           len(self.query_master_indices))

    def _save_to_cache(self):
        """Save dataset to cache."""
        data_to_cache = {
            'uniprot_to_internal_id': self.uniprot_to_internal_id,
            'internal_id_to_proteome_seq': self.internal_id_to_proteome_seq,
            'all_unique_input_ids': self.all_unique_input_ids,
            'all_unique_attention_masks': self.all_unique_attention_masks,
            'query_master_indices': self.query_master_indices,
            'ground_truth_map_query_id_to_partner_ids': self.ground_truth_map_query_id_to_partner_ids,
            'total_possible_query_uniprots': self.total_possible_query_uniprots,
        }
        pickle_save(data_to_cache, self.cache_file)
        self.logger.info(f"Saved ProteomeRankingDataset to cache: {self.cache_file}")

    def _load_and_process_dataset(self):
        """Load and process the dataset from scratch."""
        self.logger.info(
            f"Building ProteomeRankingDataset with {self.organism} proteome and {self.cfg.data} queries...")

        # 1. Load Proteome
        proteome_fasta_path = self.proteome_fasta_paths.get(self.organism)
        if not proteome_fasta_path or not os.path.exists(proteome_fasta_path):
            raise FileNotFoundError(f"Proteome FASTA for {self.organism} not at {proteome_fasta_path}")
        fasta_uniprot_to_seq_map = read_proteome_fasta(proteome_fasta_path, self.logger)

        # 2. Process Proteome for Unique Sequences and Internal IDs
        unique_proteome_sequences_list = sorted(list(set(fasta_uniprot_to_seq_map.values())))
        self.internal_id_to_proteome_seq = {i: seq for i, seq in enumerate(unique_proteome_sequences_list)}
        proteome_seq_to_internal_id = {seq: i for i, seq in self.internal_id_to_proteome_seq.items()}

        self.uniprot_to_internal_id = {}
        for uniprot_id, seq_str in fasta_uniprot_to_seq_map.items():
            if seq_str in proteome_seq_to_internal_id:  # Ensure sequence made it to unique list
                self.uniprot_to_internal_id[uniprot_id] = proteome_seq_to_internal_id[seq_str]

        num_unique_proteome_proteins = len(unique_proteome_sequences_list)
        self.logger.info(f"Processed {num_unique_proteome_proteins} unique sequences from {self.organism} proteome.")

        # 3. Tokenize Unique Proteome Sequences
        self.logger.info(
            f"Tokenizing {num_unique_proteome_proteins} unique proteome sequences "
            f"(max_length={self._max_length_for_processing})...")
        
        all_input_ids_list, all_attention_masks_list = [], []
        for i in tqdm(range(num_unique_proteome_proteins), desc=f"Tokenizing {self.organism} proteome"):
            seq_str = self.internal_id_to_proteome_seq[i]
            tokenized = self._tokenizer_for_processing(
                seq_str, 
                truncation=True, 
                padding='max_length',
                max_length=self._max_length_for_processing, 
                return_tensors='pt'
            )
            all_input_ids_list.append(tokenized.input_ids.squeeze(0))
            all_attention_masks_list.append(tokenized.attention_mask.squeeze(0))
            
        self.all_unique_input_ids = torch.stack(all_input_ids_list)
        self.all_unique_attention_masks = torch.stack(all_attention_masks_list)
        self.logger.info(f"Tokenized proteome tensors created.")

        # 4. Load Query Dataset and Map to Proteome Internal IDs
        self.logger.info(f"Loading query dataset '{self.cfg.data}' for queries/GT...")
        query_hf_dataset_dict = load_hf_dataset_from_local_if_possible(
            self.query_hf_dataset_cfg.hf_path,
            self.query_hf_dataset_cfg.get('local_path'),
            token=self.cfg.get('hf_access_token'),
        )
        test_split_data = query_hf_dataset_dict.get('test')
        if not test_split_data:
            raise ValueError(f"Test split not in query dataset '{self.cfg.data}'.")

        potential_query_internal_ids_set = set()
        self.ground_truth_map_query_id_to_partner_ids = {}

        for item in tqdm(test_split_data, desc=f"Processing test split of {self.cfg.data} for GT"):
            label = item.get('label')
            pair_id_str = item.get('id')  # Expected format "uniprot1_uniprot2"

            if not (label == 1 and isinstance(pair_id_str, str) and '_' in pair_id_str):
                continue

            try:
                uniprot_id1, uniprot_id2 = pair_id_str.split('_', 1)
            except ValueError:
                self.logger.debug(f"Could not parse pair ID: {pair_id_str}. Skipping.")
                continue

            # Process both directions: uniprot_id1 -> uniprot_id2 and uniprot_id2 -> uniprot_id1
            query_internal_id1 = self.uniprot_to_internal_id.get(uniprot_id1)
            partner_internal_id2 = self.uniprot_to_internal_id.get(uniprot_id2)

            # Both proteins must exist in our proteome map
            if query_internal_id1 is not None and partner_internal_id2 is not None:
                # Add both proteins to potential query set
                potential_query_internal_ids_set.add(query_internal_id1)
                potential_query_internal_ids_set.add(partner_internal_id2)

                # Add bidirectional relationships
                if query_internal_id1 not in self.ground_truth_map_query_id_to_partner_ids:
                    self.ground_truth_map_query_id_to_partner_ids[query_internal_id1] = set()
                self.ground_truth_map_query_id_to_partner_ids[query_internal_id1].add(partner_internal_id2)

                if partner_internal_id2 not in self.ground_truth_map_query_id_to_partner_ids:
                    self.ground_truth_map_query_id_to_partner_ids[partner_internal_id2] = set()
                self.ground_truth_map_query_id_to_partner_ids[partner_internal_id2].add(query_internal_id1)

        # Set total possible queries
        self.total_possible_query_uniprots = len(potential_query_internal_ids_set)

        query_master_indices_list_full = sorted(list(potential_query_internal_ids_set))

        # Sample subset if requested
        num_ranking_eval = self.cfg.get('num_ranking_eval', -1)
        if num_ranking_eval > 0 and len(query_master_indices_list_full) > num_ranking_eval:
            rng = np.random.default_rng(seed=0)
            permutated_samples = rng.permutation(len(query_master_indices_list_full))
            permutated_samples = [query_master_indices_list_full[i] for i in permutated_samples]
            self.query_master_indices = permutated_samples[:num_ranking_eval]
        else:
            self.query_master_indices = query_master_indices_list_full

        self.logger.info(
            f"Selected {len(self.query_master_indices)} query protein internal IDs (from {self.cfg.data} test set, "
            f"mapped to {self.organism} proteome) out of {self.total_possible_query_uniprots} total unique proteins "
            f"with interactions found in proteome.")

    @property
    def num_unique_proteins(self) -> int:
        """Get the number of unique proteins in the proteome."""
        return len(self.internal_id_to_proteome_seq)

    def get_query_indices(self) -> List[int]:
        """Get the list of query protein indices."""
        return self.query_master_indices

    def get_ground_truth_partners(self, query_idx: int) -> Set[int]:
        """Get the set of ground truth partner indices for a query protein."""
        return self.ground_truth_map_query_id_to_partner_ids.get(query_idx, set())

    def get_tokenized_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the tokenized input_ids and attention_masks for all proteins."""
        return self.all_unique_input_ids, self.all_unique_attention_masks


def compute_proteome_embeddings(cfg, model, dataset: ProteomeRankingDataset, logger, rebuild: bool = False):
    """Compute or load protein embeddings for the proteome dataset."""
    device = get_device(cfg)
    use_sorf = cfg.get("use_sorf", True)
    emb_suffix = "_sorf" if use_sorf else ""
    emb_path = os.path.join(cfg.dirs.output, f"proteome_{dataset.organism}_embeddings{emb_suffix}.npy")

    if os.path.exists(emb_path) and not rebuild:
        logger.info(f"Loading existing proteome embeddings from {emb_path}")
        return np.load(emb_path)

    logger.info("Rebuilding proteome embeddings..." if os.path.exists(emb_path) else "Computing proteome embeddings...")
    num_proteins = dataset.num_unique_proteins
    embeddings_list = []
    eval_bsz = 64
    all_unique_input_ids, all_unique_attention_masks = dataset.get_tokenized_sequences()

    for i in tqdm(range(0, num_proteins, eval_bsz), desc="Embedding proteome"):
        end_idx = min(i + eval_bsz, num_proteins)
        input_ids_batch = all_unique_input_ids[i:end_idx].to(device)
        attention_mask_batch = all_unique_attention_masks[i:end_idx].to(device)
        tokens_batch = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
        with torch.no_grad():
            protein_embs = model.get_protein_embedding(tokens_batch)
        embeddings_list.extend(protein_embs.float().cpu().numpy())

    embeddings_array = np.stack(embeddings_list, axis=0)
    np.save(emb_path, embeddings_array)
    logger.info(f"Saved proteome embeddings to {emb_path}")
    return embeddings_array


def evaluate_proteome_retrieval(cfg, model, tokenizer, logger):
    use_sorf = cfg.get("use_sorf", False)
    sorf_sigma = getattr(model, "kernel_sigma", cfg.get("sorf_sigma", cfg.get("sigma", 1.0)))
    proteome_paths = {"human": cfg.proteome_paths.human, "yeast": cfg.proteome_paths.yeast}

    dataset = ProteomeRankingDataset(
        cfg=cfg,
        logger=logger,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        proteome_fasta_paths=proteome_paths,
    )
    embeddings_array = compute_proteome_embeddings(cfg, model, dataset, logger, rebuild=True)

    hnsw_config = {
        "space": "cosine",
        "ef_construction": 100,
        "ef_search": 100,
        "max_neighbors": 16,
        "num_threads": os.cpu_count(),
    }
    sorf_suffix = "_sorf" if use_sorf else ""
    db_name = f"proteome_{dataset.organism}_collection{sorf_suffix}_D{cfg.rff_dim}"

    num_proteins = embeddings_array.shape[0]
    documents = [f"Protein {i}" for i in range(num_proteins)]
    ids = [str(i) for i in range(num_proteins)]
    embeddings_list = embeddings_array.tolist()

    client = chromadb.Client(Settings(anonymized_telemetry=False))
    logger.info("Start creating ChromaDB collection.")
    collection = client.get_or_create_collection(name=db_name, configuration={"hnsw": hnsw_config})

    transformation_info = " with SORF transformation" if use_sorf else ""
    logger.info(f"Created ChromaDB collection{transformation_info} with optimized HNSW parameters:")
    logger.info(f"  - ef_construction: {hnsw_config['ef_construction']}")
    logger.info(f"  - ef_search: {hnsw_config['ef_search']}")
    logger.info(f"  - max_neighbors: {hnsw_config['max_neighbors']}")
    logger.info(f"  - num_threads: {hnsw_config['num_threads']}")
    if use_sorf:
        logger.info(f"  - SORF enabled: sigma={sorf_sigma}, seed={cfg.get('seed', 42)}")

    batch_size = client.get_max_batch_size()
    for i in tqdm(range(0, num_proteins, batch_size), desc="Upserting embeddings"):
        collection.upsert(
            documents=documents[i:i + batch_size],
            ids=ids[i:i + batch_size],
            embeddings=embeddings_list[i:i + batch_size],
        )

    recall_percentages = cfg.get("ranking_recall_percentages", [1, 5, 10])
    recall_counters = {f"recall@{p}": [0, 0] for p in recall_percentages}
    query_master_indices = dataset.get_query_indices()
    logger.info(f"Evaluating {len(query_master_indices)} queries against {num_proteins} unique proteome sequences")

    retrieval_times_per_p = {p: [] for p in recall_percentages}
    start_time = time.time()
    total_pairs = 0

    for query_idx in tqdm(query_master_indices, desc="Evaluating proteome retrieval"):
        true_partner_ids = dataset.get_ground_truth_partners(query_idx)
        if not true_partner_ids:
            continue
        total_pairs += len(true_partner_ids)
        query_emb = embeddings_array[query_idx]

        for p_val in recall_percentages:
            k = math.ceil((p_val / 100.0) * num_proteins)
            query_start_time = time.time()
            results = collection.query(query_embeddings=[query_emb.tolist()], n_results=k + 1)
            retrieval_times_per_p[p_val].append(time.time() - query_start_time)

            retrieved_ids_str = results["ids"][0]
            retrieved_ids = [int(id_str) for id_str in retrieved_ids_str if id_str.isdigit()]
            retrieved_ids = [pid for pid in retrieved_ids if pid != query_idx]
            current_recall_result = calculate_recall_at_top_x_percent(
                retrieved_ids, true_partner_ids, [p_val], num_proteins, logger
            )
            for key, (hits, total) in current_recall_result.items():
                recall_counters[key][0] += hits
                recall_counters[key][1] += total

    final_metrics = {}
    for p_val in recall_percentages:
        recall_key = f"recall@{p_val}"
        hits, total = recall_counters[recall_key]
        recall = hits / total if total > 0 else 0.0
        final_metrics[recall_key] = recall

        times_for_p = retrieval_times_per_p[p_val]
        avg_time_for_p = sum(times_for_p) / len(times_for_p) if times_for_p else 0
        total_possible_queries = dataset.total_possible_query_uniprots
        estimated_time_for_p = avg_time_for_p * total_possible_queries
        final_metrics[f"avg_time_per_query_p{p_val}_seconds"] = avg_time_for_p
        final_metrics[f"estimated_time_for_recall_top_{p_val}_percent"] = estimated_time_for_p

        logger.info(f"Recall@top{p_val}% (proteome): {recall:.4f} ({hits}/{total} pairs)")
        logger.info(f"Average time per query for top {p_val}%: {avg_time_for_p:.4f} seconds")
        logger.info(
            f"Estimated time for all {total_possible_queries} queries at top {p_val}%: {estimated_time_for_p:.2f} seconds"
        )

    final_metrics["total_proteome_proteins"] = num_proteins
    final_metrics["processed_queries"] = len(query_master_indices)
    final_metrics["total_pairs_evaluated"] = total_pairs
    final_metrics["total_evaluation_time"] = time.time() - start_time
    final_metrics["use_sorf"] = use_sorf
    if use_sorf:
        final_metrics["sorf_sigma"] = float(sorf_sigma)
        final_metrics["sorf_seed"] = cfg.get("seed", 42)

    logger.info(f"Total proteins in {dataset.organism} proteome: {num_proteins}")
    logger.info(f"Total protein pairs evaluated: {total_pairs}")
    logger.info(f"Total evaluation time: {final_metrics['total_evaluation_time']:.2f} seconds")
    if use_sorf:
        logger.info(f"SORF transformation was applied with sigma={sorf_sigma}")

    logger.log_metrics(final_metrics, level="info")
    return final_metrics
