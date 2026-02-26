<div align="center">

# Fast Proteome-Scale Protein Interaction Retrieval via Residue-Level Factorization #

[![pytorch](https://img.shields.io/badge/PyTorch_2.1+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)

</div>

Original PyTorch implementation of the ICLR26 paper "Fast Proteome-Scale Protein Interaction Retrieval via Residue-Level Factorization".

## Abstract ##
Protein-protein interactions (PPIs) are mediated at the residue level. Most sequence-based PPI models consider residue-residue interactions across two proteins, which can yield accurate interaction scores but are too slow to scale. At proteome scale, identifying candidate PPIs requires evaluating nearly \textit{all possible protein pairs}. For $N$ proteins of average length $L$, exhaustive all-against-all search requires $\mathcal{O}(N^2L^2)$ computation, rendering conventional approaches computationally impractical. We introduce RaftPPI, a scalable framework that approximates residue-level PPI modeling while enabling efficient large-scale retrieval. RaftPPI represents residue interactions with a Gaussian kernel, approximated efficiently via structured random Fourier features, and applies a low-rank factorized attention mechanism that admits pooling into a compact embedding per protein. Each protein is encoded once into an indexable embedding, allowing approximate nearest-neighbor search to replace exhaustive pairwise scoring, reducing proteome-wide retrieval from \textit{months} to \textit{minutes} on a single GPU. On the human proteome with the D-SCRIPT dataset, RaftPPI retrieves the top 20\% pairs from ~200M candidate pairs in 5.7 GPU minutes, or 3.3 Intel Xeon6 6980P CPU minutes, covering 75.1\% of the true interacting pairs,
compared to 4.9 GPU months for the best prior method (61.2\%). Across seven benchmarks with sequence- and degree-controlled splits, RaftPPI achieves state-of-the-art PPI classification and retrieval performance, while enabling residue-aware, retrieval-friendly screening at proteome scale.

## Overview ##

![RaftPPI Model](asset/RaftPPI_model.png)

RaftPPI supports both standalone classification and large-scale retrieval settings.
Experiments and configurations are managed with Hydra and can be switched via simple
CLI flags.

## Environment Setup ##

Our experiments run on GPU or CPU.

### uv (recommended)

```bash
# Install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
uv sync --python 3.10
source .venv/bin/activate
```

After activating the uv environment, use `python` commands directly.

## File Structure ##

```
├── .project-root
├── LICENSE
├── README.md
├── checkpoints
├── configs
│   ├── main.yaml
│   └── data.yaml
├── src
│   ├── raft
│   │   ├── data.py
│   │   ├── model.py
│   │   └── proteome_rff_retrieval.py
│   ├── train_raft.py
│   ├── utils
│   └── ...
├── data
└── asset
```
`data_cache/` and `output/` are generated at runtime.

Important notes: All reported results are obtained with DeepSpeed 0.17.2 (ZeRO Stage 3) on ampere and lovelace GPUs (e.g. L40S, A100). The code has not been tested with newer DeepSpeed versions and older Nvidia GPUs.

## Reproduce Classification and Retrieval ##
### Evaluate From Released Checkpoints (Inference Only) ##

Use the Hydra multirun command below to recover the classification and retrieval process:

```bash
cd path/to/the/project/directory/
python src/train_raft.py -m data=guo,du,huang,dscript,pan,richoux,gold use_wandb=false
```
Alternatively, you can also run one dataset at a time and iterate over dataset.

```bash
python src/train_raft.py -m data=guo use_wandb=false
```

### Evaluate From Released Checkpoints (Inference Only) ##

Use Hydra multirun to evaluate all released datasets with their corresponding checkpoints:

```bash
python src/train_raft.py -m data=guo,du,huang,dscript,pan,richoux,gold pretrained_dir='checkpoints/${data}' max_steps=0 use_wandb=false
```

Explanation:
- `data=...` sets the current dataset for each run.
- `pretrained_dir='checkpoints/${data}'` must point to a `.bin` checkpoint file or a directory containing `pytorch_model.bin`.
- `max_steps=0` skips finetuning and runs evaluation only (classification + proteome retrieval recall by default).

## Configuration ##

- **Hydra configs**: Located under `configs/` (`main.yaml`, `data.yaml`, `model.yaml`).
- **Experiment control**: Modify settings via CLI, e.g., `python src/train_raft.py data=pan max_steps=1000 lr=5e-5`.
- **Outputs**: Logs, checkpoints, and artifacts are stored under `output/`.
- **Wandb**: Report on wandb: specify `wandb.entity=your_wandb_entity wandb.project=your_wandb_project use_wandb=true` to log on wandb.

## Citation ##

If you find this codebase useful in your research, please cite:

```bibtex
@inproceedings{zhao2026raftppi,
  title = {Fast Proteome-Scale Protein Interaction Retrieval via Residue-Level Factorization},
  author = {Zhao, Jianan and Zhan, Zhihao and Chaudhary, Narendra and Yuan, Xinyu and Zhang, Zuobai and Cong, Qian and Zhou, Jian and Misra, Sanchit and Tang, Jian},
  booktitle = {International Conference on Learning Representations},
  year = {2026}
}
```
