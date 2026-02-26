import numpy as np
import os
import random
import rootutils
import sys
import torch
import wandb
from datetime import datetime
from omegaconf import OmegaConf
from uuid import uuid4

from .config import save_config, print_important_cfg, get_important_cfg
from .log import ExpLogger, logger

root = rootutils.find_root(__file__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_unique_id(cfg):
    """Generate a Unique ID (UID) for (1) File system (2) Communication between submodules
    By default, we use time and UUID4 as UID. UIDs could be overwritten by wandb or UID specification.
    """
    #
    if cfg.get("uid") is not None and cfg.wandb.id is not None:
        assert cfg.get("uid") == cfg.wandb.id, "Confliction: Wandb and uid mismatch!"
    cur_time = datetime.now().strftime("%b%-d-%-H:%M-")
    given_uid = cfg.wandb.id or cfg.get("uid")
    uid = given_uid if given_uid else cur_time + str(uuid4()).split("-")[0]
    return uid


def setup_offline_mode(cfg, logger):
    if cfg.get("offline_mode", False):
        # Set HuggingFace to work offline
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        logger.critical("Offline mode enabled. HuggingFace will use local files only.")
        # If offline mode is enabled, force wandb to offline mode too
        logger.critical("Setting wandb to offline mode due to offline_mode=True")
        cfg.wandb.mode = "offline"


def init_experiment(cfg, init_wandb=True):
    # Prevent ConfigKeyError when accessing non-existing keys
    OmegaConf.set_struct(cfg, False)
    cfg.cmd = 'python ' + ' '.join(sys.argv)

    if torch.cuda.is_available():
        cfg.device_type = "GPU"
    else:
        cfg.device_type = "XPU" if hasattr(torch, 'xpu') and torch.xpu.is_available() else "CPU"

    if 'pd_batch_size' in cfg:  # Per device batch size
        cfg.eq_batch_size = cfg.pd_batch_size * cfg.get("grad_acc_steps", 1)

    if init_wandb:
        wandb_init(cfg)

    # Path related
    cfg.uid = generate_unique_id(cfg)
    for directory in cfg.dirs.values():
        os.makedirs(directory, exist_ok=True)

    cfg_out_file = os.path.join(cfg.dirs.output, "hydra_cfg.yaml")
    save_config(cfg, cfg_out_file, as_global=True)
    # Initialize logger
    exp_logger = ExpLogger(cfg)
    print_important_cfg(cfg, exp_logger.critical)
    exp_logger.save_file_to_wandb(cfg_out_file, base_path=cfg.dirs.output, policy="now")
    exp_logger.warning(f'Running\n{cfg.cmd}\n Running on {cfg.device_type}\n'
                       f'output_dir={cfg.dirs.output}')
    set_seed(cfg.seed)
    setup_offline_mode(cfg, exp_logger)
    return cfg, exp_logger


def get_device(cfg):
    if cfg.device_type == "XPU" and torch.xpu.is_available():
        device = "xpu:0"
        logger.info("Using single XPU: xpu:0")
    elif cfg.device_type == "GPU" and torch.cuda.is_available():
        device = "cuda:0"
        logger.info("Using single GPU: cuda:0")
    else:
        device = "cpu"
        logger.info("Using CPU")
    return device


def wandb_init(cfg) -> None:
    os.environ["WANDB_WATCH"] = "false"  # No gradients are watched
    if cfg.use_wandb:
        wandb_tags = cfg.wandb.tags
        mode = cfg.wandb.get('mode', 'online') if not cfg.get('offline_mode', False) else 'offline'
        imp_cfg = get_important_cfg(cfg)
        logger.critical(f'Creating WANDB session {mode=}')
        if cfg.wandb.id is None:
            # First time running, create new wandb
            os.makedirs(cfg.dirs.wandb_cache, exist_ok=True)
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                dir=cfg.dirs.wandb_cache,
                reinit=True,
                config=imp_cfg,
                name=cfg.wandb.name,
                tags=wandb_tags,
                mode=mode,
            )
        else:  # Resume from previous run
            logger.critical(f"Resume from previous wandb run {cfg.wandb.id}")
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                reinit=True,
                resume="must",
                id=cfg.wandb.id,
                mode=mode,
            )
            cfg.wandb.is_master_process = False  # Running as a sub_process
        if mode == "online":
            cfg.wandb.id, cfg.wandb.name, cfg.wandb.url = (
                wandb.run.id,
                wandb.run.name,
                wandb.run.url,
            )
        if mode == 'offline':
            logger.critical(f'Wandb local mode use command to sync:\n'
                            f'wandb sync --include-offline {cfg.dirs.wandb_cache}wandb/offline-*')
        step_metric = cfg.wandb.get('step_metric', None)
        if step_metric:
            wandb.run.define_metric(step_metric)
        wandb.run.define_metric("*", step_metric=step_metric, step_sync=True)
    else:
        # If wandb not already initialized, set all wandb settings to None.
        os.environ["WANDB_DISABLED"] = "true"
        cfg.use_wandb = False
        cfg.wandb.id, cfg.wandb.name, cfg.wandb.url = None, None, None
