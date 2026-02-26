import os
import shutil
from typing import Dict

import deepspeed
import hydra
import rootutils
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from src.raft.model import RaftModel
from src.raft.data import PPITripletCollator, load_single_dscript_dataset
from src.raft.proteome_rff_retrieval import evaluate_proteome_retrieval, evaluate_ranking
from src.utils.distributed import empty_cache, initialize_deepspeed_and_float_type
from src.utils.experiment import init_experiment
from src.utils.log import timer

###########################################################
# Evaluation Functions (Moved and Adapted)
###########################################################
def resolve_pretrained_bin_path(pretrained_dir: str):
    if pretrained_dir is None or pretrained_dir == "None":
        return None

    base_path = os.path.normpath(str(pretrained_dir))

    if os.path.isfile(base_path):
        return base_path if base_path.endswith(".bin") else None

    candidate_bin_path = os.path.join(base_path, "pytorch_model.bin")
    if os.path.isfile(candidate_bin_path):
        return candidate_bin_path

    return None


def load_model_weights_from_bin(model_engine, checkpoint_file: str, logger):
    try:
        try:
            raw_state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        except TypeError:
            raw_state_dict = torch.load(checkpoint_file, map_location="cpu")

        if not isinstance(raw_state_dict, dict):
            logger.warning(f"Invalid checkpoint format in {checkpoint_file}. Expected a dict-like state_dict.")
            return False

        state_dict = raw_state_dict.get("state_dict", raw_state_dict)
        if not isinstance(state_dict, dict):
            logger.warning(f"Invalid 'state_dict' field in {checkpoint_file}.")
            return False

        load_result = model_engine.module.load_state_dict(state_dict, strict=False)
        if hasattr(load_result, "missing_keys"):
            missing_keys = list(load_result.missing_keys)
            unexpected_keys = list(load_result.unexpected_keys)
        else:
            missing_keys, unexpected_keys = load_result

        matched_keys = len(set(model_engine.module.state_dict().keys()) & set(state_dict.keys()))
        if matched_keys == 0:
            logger.warning(f"No overlapping parameter keys found when loading {checkpoint_file}.")
            return False

        logger.info(f"Loaded {matched_keys} parameter tensors from {checkpoint_file}.")
        if missing_keys:
            logger.warning(
                f"Missing {len(missing_keys)} keys when loading pretrained weights. "
                f"Examples: {missing_keys[:5]}"
            )
        if unexpected_keys:
            logger.warning(
                f"Unexpected {len(unexpected_keys)} keys in pretrained weights. "
                f"Examples: {unexpected_keys[:5]}"
            )
        return True
    except Exception as exc:
        logger.warning(f"Failed to load pretrained weights from {checkpoint_file}: {exc}")
        return False


def save_model_weights_to_bin(model_engine, checkpoint_root: str, step: int, logger):
    step_dir = os.path.join(checkpoint_root, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)
    step_path = os.path.join(step_dir, "pytorch_model.bin")

    torch.save(model_engine.module.state_dict(), step_path)

    latest_dir = os.path.join(checkpoint_root, "latest")
    os.makedirs(latest_dir, exist_ok=True)
    latest_path = os.path.join(latest_dir, "pytorch_model.bin")
    shutil.copy2(step_path, latest_path)

    logger.info(f"Saved model weights to {step_path}")
    logger.info(f"Updated latest model weights at {latest_path}")


@torch.no_grad()
def evaluate_single_dataset(model,
                            ds_name: str,
                            protein_pairs,
                            eval_bsz: int):

    pos_pairs_data, neg_pairs_data = protein_pairs
    all_pairs_with_labels = (
        [((r_tok_d, l_tok_d), 1) for r_tok_d, l_tok_d in pos_pairs_data]
        + [((r_tok_d, l_tok_d), 0) for r_tok_d, l_tok_d in neg_pairs_data]
    )

    all_scores_gathered = []
    all_labels_gathered = []

    for start_idx in tqdm(
        range(0, len(all_pairs_with_labels), eval_bsz),
        desc=f"Evaluating {ds_name}",
        unit="batch",
        disable=False,
        leave=False,
    ):
        batch_tuples = all_pairs_with_labels[start_idx: start_idx + eval_bsz]

        current_batch_r_tokenized = [item[0][0] for item in batch_tuples]
        current_batch_l_tokenized = [item[0][1] for item in batch_tuples]
        current_batch_labels_list = [item[1] for item in batch_tuples]

        target_device = model.device

        r_tokens_padded = model.module.tokenizer.pad(current_batch_r_tokenized, padding=True, return_tensors="pt")
        l_tokens_padded = model.module.tokenizer.pad(current_batch_l_tokenized, padding=True, return_tensors="pt")

        r_tokens_dev = {k: v.to(target_device) for k, v in r_tokens_padded.items()}
        l_tokens_dev = {k: v.to(target_device) for k, v in l_tokens_padded.items()}
        labels_dev = torch.tensor(current_batch_labels_list, dtype=torch.float32).to(target_device)

        prot_scores, _ = model(r_tokens_dev, l_tokens_dev, labels_dev, mode='inference')

        all_scores_gathered.extend(prot_scores.detach().cpu().tolist())
        all_labels_gathered.extend(current_batch_labels_list)

        del prot_scores, r_tokens_dev, l_tokens_dev, labels_dev, r_tokens_padded, l_tokens_padded

    if model.device.type == 'cuda':
        empty_cache("cuda")
    final_scores = all_scores_gathered
    final_labels = all_labels_gathered
    if not final_scores:
        return {'auc': 0.0}
    results = evaluate_ranking(final_labels, final_scores)
    return results


@torch.no_grad()
def evaluate_classification(model,
                            cfg: DictConfig,
                            data_dict,
                            logger,
                            metrics: Dict,
                            skipped_splits=("test",),
                            ):
    max_eval_samples = cfg.get('max_eval_samples')

    is_training_before_eval = model.training
    model.eval()
    if model.device.type == 'cuda': empty_cache("cuda")

    eval_metrics_results = {}
    for ds_name, ds_obj in data_dict.items():
        if any(skipped_split in ds_name for skipped_split in skipped_splits):
            continue

        pos_pairs_tokenized, neg_pairs_tokenized = ds_obj.protein_pairs

        current_max_samples = max_eval_samples
        if ("train" in ds_name) and (current_max_samples is None):
            current_max_samples = cfg.get('max_train_eval_samples', 100)

        if current_max_samples is not None:
            pos_pairs_tokenized = pos_pairs_tokenized[:current_max_samples]
            neg_pairs_tokenized = neg_pairs_tokenized[:current_max_samples]

        if not pos_pairs_tokenized and not neg_pairs_tokenized:
            logger.warning(f"No protein pairs to evaluate for {ds_name} after sampling. Skipping.")
            continue

        res = evaluate_single_dataset(
            model,
            ds_name,
            (pos_pairs_tokenized, neg_pairs_tokenized),
            cfg.eval_bsz,
        )

        for m, v in res.items():
            eval_metrics_results[f"{ds_name}_{m}"] = v

    metrics.update(eval_metrics_results)
    logger.log_metrics(metrics, level="info", step=model.global_steps)

    if is_training_before_eval: model.train()
    if model.device.type == 'cuda': empty_cache("cuda")

    return metrics


###########################################################
# Main Training Loop (Adapted)
###########################################################
def train_model(cfg: DictConfig, model_engine, data_dict, logger):
    pretrained_bin_path = resolve_pretrained_bin_path(cfg.get("pretrained_dir"))
    if pretrained_bin_path is None:
        if cfg.pretrained_dir is None or cfg.pretrained_dir == "None":
            logger.info("No pretrained_dir specified. Training from base model.")
        else:
            logger.warning(
                f"Could not find a checkpoint in {cfg.pretrained_dir}. "
                "Expected a .bin path or a directory containing pytorch_model.bin. "
                "Training from base model."
            )
    else:
        success = load_model_weights_from_bin(model_engine, pretrained_bin_path, logger)
        model_engine.global_steps = 0
        if success:
            logger.info(f"Loaded pre-trained model weights from {pretrained_bin_path}.")
        else:
            logger.warning(f"Failed to load pre-trained model from {pretrained_bin_path}.")

    checkpoint_dir = os.path.join(cfg.dirs.output, "checkpoints")

    # Instantiate collator with the tokenizer from the model (engine.module.tokenizer)
    collator = PPITripletCollator(max_length=cfg.max_length, tokenizer=model_engine.module.tokenizer)
    triplets = data_dict['train'].to_ppi_triplets()
    train_loader = DataLoader(triplets, batch_size=cfg.pd_batch_size, shuffle=True,
                              collate_fn=collator, drop_last=True, num_workers=cfg.get('num_workers', 2))

    epoch = 0
    accumulated_loss_for_period = 0.0
    steps_in_period = 0

    pbar = tqdm(
        total=cfg.max_steps,
        initial=int(model_engine.global_steps),
        desc=f"Epoch {epoch + 1} Training",
        unit="step",
        disable=False,
        mininterval=cfg.get('pbar_mininterval', 3)
    )

    while model_engine.global_steps < cfg.max_steps:
        epoch += 1
        model_engine.train()  # Ensure model is in training mode
        pbar.set_description(f"Epoch {epoch} Training (Overall Steps: {model_engine.global_steps}/{cfg.max_steps})")

        # Reset accumulators if not evaluating frequently within an epoch
        if not (cfg.eval_steps and cfg.eval_steps < len(train_loader)):
            accumulated_loss_for_period = 0.0
            steps_in_period = 0

        for batch in train_loader:
            if model_engine.global_steps >= cfg.max_steps:
                break

            target_device = model_engine.device
            r_tokens = {k: v.to(target_device) for k, v in batch["r_tokens"].items()}
            l_tokens = {k: v.to(target_device) for k, v in batch["l_tokens"].items()}
            labels = batch["labels"].to(target_device)

            _, prot_loss = model_engine(r_tokens, l_tokens, labels, mode='training')

            # Backward pass and step
            model_engine.backward(prot_loss)
            model_engine.step()
            empty_cache("cuda")
            accumulated_loss_for_period += prot_loss.item()
            steps_in_period += 1

            # Trigger logging/eval/checkpoint only when global step advances.
            if model_engine.is_gradient_accumulation_boundary():
                current_step = model_engine.global_steps
                pbar.update(1)
                current_avg_loss_for_period = accumulated_loss_for_period / steps_in_period if steps_in_period > 0 \
                    else 0
                pbar.set_postfix(loss=f"{prot_loss.item():.4f}",
                                 avg_loss_period=f"{current_avg_loss_for_period:.4f}",
                                 epoch=epoch,
                                 step=current_step)
                pbar.set_description(
                    f"Epoch {epoch} Training (Overall Steps: {current_step}/{cfg.max_steps})")

                if cfg.eval_steps and current_step > 1 and (current_step % cfg.eval_steps == 0):
                    avg_loss_to_log = accumulated_loss_for_period / steps_in_period if steps_in_period > 0 else 0
                    eval_metrics = {
                        "epoch": epoch,
                        "train_loss_period_avg": avg_loss_to_log,
                        "lr": model_engine.optimizer.param_groups[0]['lr']
                    }
                    with timer(f"evaluation at step {current_step}", logger.info):
                        evaluate_classification(model_engine, cfg, data_dict, logger, eval_metrics)

                    accumulated_loss_for_period = 0.0
                    steps_in_period = 0
                    model_engine.train()  # Ensure back to train mode after eval

                if cfg.save_steps and current_step > 0 and (current_step % cfg.save_steps == 0):
                    logger.info(f"Saving checkpoint at step {current_step}")
                    save_model_weights_to_bin(model_engine, checkpoint_dir, current_step, logger)

        # Log end-of-epoch average loss if not captured by eval_steps alignment
        if steps_in_period > 0 and not (cfg.eval_steps and (len(train_loader) % cfg.eval_steps == 0)):
            avg_loss_epoch_end = accumulated_loss_for_period / steps_in_period
            logger.log_metrics({
                "epoch_train_loss_avg": avg_loss_epoch_end,
                "epoch": epoch
            }, level="info", step=model_engine.global_steps)
            if not (cfg.eval_steps and cfg.eval_steps < len(train_loader)):
                accumulated_loss_for_period = 0.0
                steps_in_period = 0

    pbar.close()
    logger.info(f"Training completed after {model_engine.global_steps} steps.")

    final_avg_loss = accumulated_loss_for_period / steps_in_period if steps_in_period > 0 else 0.0
    final_eval_metrics = {
        "step": model_engine.global_steps,
        "epoch": epoch,
        "prot_loss_final_training_avg": final_avg_loss,
        "lr": model_engine.optimizer.param_groups[0]['lr']
    }
    logger.info("Performing final evaluation on validation and test splits...")
    evaluate_classification(
        model_engine,
        cfg,
        data_dict,
        logger,
        final_eval_metrics,
        skipped_splits=("train",),
    )

    final_checkpoint_dir = os.path.join(cfg.dirs.output, "final_checkpoint")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    final_checkpoint_path = os.path.join(final_checkpoint_dir, "pytorch_model.bin")
    torch.save(model_engine.module.state_dict(), final_checkpoint_path)
    logger.info(f"Final model weights saved to {final_checkpoint_path}")


###########################################################
# Hydra Main (Adapted)
###########################################################
@timer()
@hydra.main(config_path=f'{root}/configs', config_name='main', version_base=None)
def main(cfg: DictConfig):
    cfg, logger = init_experiment(cfg)
    float_dtype = initialize_deepspeed_and_float_type(cfg)

    tmp_hf_ckpt = getattr(cfg, 'local_hf_ckpt', cfg.hf_ckpt) if cfg.get('offline_mode', False) else cfg.hf_ckpt
    tokenizer = AutoTokenizer.from_pretrained(tmp_hf_ckpt, trust_remote_code=True)
    logger.info(f"Using tokenizer from {tmp_hf_ckpt} for initial data loading.")
    data_dict = load_single_dscript_dataset(cfg, logger, tokenizer)

    # Model now takes cfg and logger
    model = RaftModel(cfg=cfg, logger=logger, float_dtype=float_dtype).to(dtype=float_dtype)
    # Model should be on CPU before DS init.

    logger.info("Optimizing all model parameters")
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    ds_config_dict = OmegaConf.to_container(cfg.deepspeed, resolve=True)

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,  # Pass optimizer directly
        config=ds_config_dict,
    )

    train_model(cfg, model_engine, data_dict, logger)
    if cfg.dataset.get('skip_proteome_ranking', False):
        logger.info(f"Skipping proteome ranking for dataset {cfg.data} as requested.")
    else:
        evaluate_proteome_retrieval(cfg, model_engine.module, tokenizer, logger)


if __name__ == "__main__":
    main()
