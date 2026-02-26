import os

import torch


def empty_cache(device_type: str) -> None:
    if device_type.upper() == "XPU" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

def initialize_deepspeed_and_float_type(cfg):
    import deepspeed

    if cfg.device_type == "GPU":
        is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)
    elif cfg.device_type == "XPU" and hasattr(torch, "xpu"):
        is_bf16_supported = torch.xpu.is_bf16_supported()
        deepspeed.init_distributed()
    else:
        is_bf16_supported = False

    cfg.rank = int(os.getenv("RANK", "0"))
    cfg.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    cfg.world_size = int(os.getenv("WORLD_SIZE", "1"))
    cfg.is_distributed = cfg.world_size > 1
    cfg.is_bf16_supported = is_bf16_supported
    if cfg.mixed_precision == "bf16" and not cfg.is_bf16_supported:
        cfg.mixed_precision = "fp16"

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    float_dtype = dtype_map[cfg.mixed_precision]

    cfg.deepspeed.bf16.enabled = cfg.mixed_precision == "bf16"
    cfg.deepspeed.fp16.enabled = not cfg.deepspeed.bf16.enabled
    return float_dtype
