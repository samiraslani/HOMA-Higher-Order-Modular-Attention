# HOMA — Higher-Order Modular Attention

HOMA (Higher-Order Modular Attention) is a sequence transformer that extends standard pairwise attention with **third-order (triadic) interactions** between sequence positions. The core contribution is a unified architecture that **fuses pairwise attention with triadic attention** through a learned MLP — capturing richer positional dependencies while remaining computationally tractable.

Standard transformers compute pairwise interactions via scaled dot-product attention:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_h}}\right)V$$

HOMA introduces a **third-order interaction term** through an additional low-rank projection $U$:

$$\text{scores}_{3D}[i,j,k] = \frac{(Q_i \odot K_j^{(w)} \odot U_k^{(w)}) \cdot \mathbf{1}}{\sqrt{d_h}}, \quad j,k \in [0,w)$$

where $j$ and $k$ index positions within a local window of size $w$ around query position $i$. The 2D and 3D outputs are fused by a small MLP, giving the model access to both pairwise and triadic structure simultaneously. Two design choices keep computation tractable:

| Technique | Effect |
|---|---|
| **Sliding-window blocking** | Restricts attention to overlapping blocks of length `block_size`, reducing complexity from $O(L^3)$ to $O(L \cdot w^2)$ |
| **Low-rank U-matrix** | Factorises $U = W_{l_u} W_{l_v}$ with inner rank $r$, cutting 3D parameters by ~97% (262 k → 8 k for $d$=512, $r$=8) |

Transfer learning is supported for training HOMA attention mechanism: pretrained 2D weights ($W_q, W_k, W_v$) can be loaded from a `blockwise2d` checkpoint and optionally frozen, so only the 3D-specific parameters ($W_{l_u}$, $W_{l_v}$, fusion MLP) are trained from scratch.

Built on the [TAPE benchmark](https://github.com/songlab-cal/tape) and evaluated on secondary structure prediction (SS3), fluorescence prediction, and stability prediction.

> **Paper:** [HOMA: Higher-Order Modular Attention for Protein Sequence Modelling](https://arxiv.org/abs/2603.11133)

---

## Contents

- [Package structure](#package-structure)
- [Installation](#installation)
- [Dataset setup](#dataset-setup)
- [Architecture](#architecture)
- [Attention mechanisms](#attention-mechanisms)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Running the examples](#running-the-examples)
- [Training output](#training-output)
- [Checkpointing](#checkpointing)
- [Efficiency tracking](#efficiency-tracking)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

## Package structure

```
HOMA-Higher-Order-Modular-Attention
├── README.md
├── config.py                       # ModelConfig, AttentionConfig, TrainingConfig
├── data/
│   ├── datasets.py                 # SecondaryStructureDataset, FluorescenceDataset, StabilityDataset
│   └── collate.py                  # collate_ss3, collate_regression
├── models/
│   ├── attention/
│   │   ├── base.py                 # AttentionBase, shared sliding-block helpers
│   │   ├── attention_2d.py         # MultiHeadAttn2D, Attn2DBlockwise, Attn2DLinformer
│   │   ├── attention_3d.py         # HOMA (main contribution), MultiHeadAttn3D
│   │   └── __init__.py             # get_attention() factory
│   ├── feedforward.py              # FeedForward
│   ├── encoder.py                  # Encoder layer
│   └── protein_transformer.py      # ProteinTransformer, PerResidueHead, GlobalRegressionHead
├── training/
│   ├── trainer.py                  # Trainer (unified loop for all tasks)
│   └── efficiency.py               # EfficiencyTracker (timing + memory)
├── evaluation/
│   └── metrics.py                  # accuracy_per_position, spearman_correlation
├── tasks/
│   ├── secondary_structure.py      # SecondaryStructureTask
│   ├── fluorescence.py             # FluorescenceTask
│   └── stability.py                # StabilityTask
├── utils/
│   ├── seed.py                     # set_seed()
│   └── checkpointing.py            # save_checkpoint, load_checkpoint
└── examples/
    ├── train_secondary_structure.py
    ├── train_fluorescence.py
    └── train_stability.py
```

---

## Installation

Requirements:
- Python >= 3.8
- PyTorch >= 1.13

```bash
git clone https://github.com/samiraslani/HOMA-Higher-Order-Modular-Attention.git
cd HOMA-Higher-Order-Modular-Attention
pip install -e .
pip install tape_proteins
```

---

## Dataset setup

This repository does not include the benchmark datasets. TAPE datasets and tokenizer are provided by the `tape_proteins` package:

```bash
pip install tape_proteins
```

Download the benchmark data following the [TAPE instructions](https://github.com/songlab-cal/tape#datasets). After downloading, place the LMDB files under the `data/` directory using the following structure:

```
data/
  secondary_structure/
    train.lmdb
    valid.lmdb
    cb513.lmdb
    casp12.lmdb
    ts115.lmdb
  fluorescence/
    train.lmdb
    valid.lmdb
    test.lmdb
  stability/
    train.lmdb
    valid.lmdb
    test.lmdb
```

If your datasets are stored elsewhere, update the dataset paths in the training scripts accordingly.

**Note**: Training and evaluation scripts will raise a `FileNotFoundError` if the expected LMDB files are not present at the paths above.

---

## Architecture

![HOMA Architecture](figures/homa_architecture.png)

*Figure 1: HOMA fuses a blockwise pairwise (2D) attention branch with a windowed triadic (3D) attention branch. The outputs of both branches are concatenated and passed through a fusion MLP to produce the final per-position representation.*

---

## Attention mechanisms

Five attention types are available via `get_attention(type, ...)` or `AttentionConfig(type=...)`:

| Type | Class | Description |
|---|---|---|
| `"plain2d"` | `MultiHeadAttn2D` | Standard scaled dot-product attention (Vaswani et al., 2017) — $O(L^2)$ |
| `"blockwise2d"` | `Attn2DBlockwise` | Pairwise attention over overlapping blocks — $O(L \cdot b^2)$ |
| `"linformer2d"` | `Attn2DLinformer` | Linformer attention — sequence length projected to low-rank dimension $k$ — $O(L \cdot k)$ |
| `"blockwise3d"` | `MultiHeadAttn3D` | Windowed triadic block attention only, no 2D branch — $O(L \cdot w^2)$ |
| `"homa"` | `HOMA` | **Main contribution** — fusion of blockwise pairwise and windowed triadic block attention |

### Complexity comparison

| Variant | Attention complexity | Memory |
|---|---|---|
| `plain2d` — Standard 2D | $O(L^2)$ | $O(L^2)$ |
| `blockwise2d` — Blockwise 2D | $O(L \cdot b^2)$ | $O(L \cdot b)$ |
| `linformer2d` — Linformer 2D | $O(L \cdot k)$ | $O(L \cdot k)$ |
| Standard 3D (naive) | $O(L^3)$ | $O(L^3)$ |
| `blockwise3d` / `homa` — Blockwise 3D | $O(L \cdot w^2)$ | $O(L \cdot w)$ |

$b$ = block size (default 30), $w$ = window size (default 7), $k$ = Linformer projection dimension, $L$ = sequence length.

---

## Quick start

### Secondary structure prediction (SS3)

```python
from tape.datasets import LMDBDataset
from tape.tokenizers import TAPETokenizer

from config import ModelConfig, AttentionConfig, TrainingConfig
from tasks.secondary_structure import SecondaryStructureTask

model_cfg = ModelConfig(d_model=512, num_layers=12, num_heads=8)
attn_cfg  = AttentionConfig(type="homa", block_size=30, stride=15, window_size=3)
train_cfg = TrainingConfig(batch_size=16, learning_rate=1e-4, epochs=10)

tokenizer = TAPETokenizer(vocab="iupac")
task = SecondaryStructureTask(model_cfg, attn_cfg, train_cfg)

model, history = task.train(
    train_lmdb=LMDBDataset("data/secondary_structure/train.lmdb"),
    val_lmdb=LMDBDataset("data/secondary_structure/valid.lmdb"),
    tokenizer=tokenizer,
    track_efficiency=True
)
```

### Transfer learning from a 2D baseline

```python
from config import AttentionConfig

attn_cfg = AttentionConfig(
    type="homa",
    block_size=30,
    stride=15,
    window_size=7,
    rank_3d=8,
    pretrained_ckpt="checkpoints/blockwise2d.pt",
    freeze_2d=False,   # set True to freeze W_q/W_k/W_v
)
```

### Using the model directly

```python
from config import ModelConfig, AttentionConfig
from models import ProteinTransformer, PerResidueHead

model_cfg = ModelConfig()
attn_cfg  = AttentionConfig(type="homa", block_size=30, stride=15)
head      = PerResidueHead(d_model=512, num_classes=3)
model     = ProteinTransformer(model_cfg, attn_cfg, head)

import torch
x = torch.randint(1, 30, (2, 128))        # (batch=2, length=128)
logits, _ = model(x, labels=torch.zeros(2, 128, dtype=torch.long))
print(logits.shape)                        # (2, 128, 3)
```

---

## Configuration

All hyperparameters are defined as Python dataclasses in `config.py`. The three dataclasses are:

- **`ModelConfig`** — architecture settings: `vocab_size`, `d_model`, `num_layers`, `num_heads`, `dim_feedforward`, `dropout`, `max_seq_length`
- **`AttentionConfig`** — attention type and its parameters: `type`, `block_size`, `stride`, `window_size`, `rank_3d`, `linformer_k`, `pretrained_ckpt`, `freeze_2d`
- **`TrainingConfig`** — optimisation settings: `batch_size`, `learning_rate`, `epochs`, `warmup_steps`, `checkpoint_dir`, `num_workers`, `device`

See `config.py` for full defaults and parameter documentation.

---

## Running the examples

```bash
# Set path to your TAPE data
export TAPE_DATA_DIR=/path/to/tape/data

# Secondary structure prediction
python examples/train_secondary_structure.py

# Fluorescence prediction
python examples/train_fluorescence.py

# Stability prediction
python examples/train_stability.py
```

Each script trains five model variants sequentially:
1. `plain2d` — standard pairwise attention baseline
2. `blockwise2d` — overlapping blocks of pairwise attention
3. `linformer2d` — Linformer low-rank attention
4. `blockwise3d` — windowed triadic block attention
5. `homa` — main contribution, optionally initialised from the `blockwise2d` checkpoint

---

## Training output

At the start of every training run the trainer prints a one-time setup summary so you can verify the configuration at a glance.

**Standard run:**

```
--------------------------------------------------
  Training Setup
--------------------------------------------------
  Attention type    : homa
  Rank              : 8
  Device            : cuda
  Epochs            : 20
  Trainable params  : 25,520,131
--------------------------------------------------
```

**With transfer learning** (`pretrained_ckpt` is set), the model first reports which checkpoint was used to initialise the 2D projection weights:

```
  Transfer learning : blockwise 2D parameters (W_q, W_k, W_v) loaded from: checkpoints/blockwise2d.pt
```

**With frozen layers** (`freeze_2d=True`), the summary additionally shows which modules are frozen and how many parameters are excluded from the optimiser:

```
  Transfer learning : blockwise 2D parameters (W_q, W_k, W_v) loaded from: checkpoints/blockwise2d.pt
  Frozen layers     : W_q, W_k, W_v  (only W_l_u, W_l_v, fusion_layer will be trained)

--------------------------------------------------
  Training Setup
--------------------------------------------------
  Attention type    : homa
  Rank              : 8
  Device            : cuda
  Epochs            : 20
  Trainable params  : 1,180,160
  Frozen params     : 786,432
  Frozen modules    : encoder_layers.0.mha.W_k, encoder_layers.0.mha.W_q, ...
--------------------------------------------------
```

Each epoch then prints a one-line progress update:

```
Epoch 1/20 | Train loss 1.2340 | Val loss 1.1890 | Val metric 0.6123
```

---

## Checkpointing

Checkpoints are saved automatically after every epoch and support seamless resumption:

```python
from training.trainer import Trainer

# Training resumes from the last checkpoint automatically
trainer = Trainer(config=train_cfg, attn_name="homa")
model, history = trainer.fit(model, train_loader, val_loader, ...)
```

Manual save / load:

```python
from utils import save_checkpoint, load_checkpoint

save_checkpoint("my_checkpoint.pt", model, optimizer, epoch=5)
ckpt = load_checkpoint("my_checkpoint.pt", model, optimizer)
```

---

## Efficiency tracking

Pass `track_efficiency=True` to any task's `train()` method to collect per-step timing and memory statistics:

```python
model, history = task.train(..., track_efficiency=True)

# history keys include:
# avg_step_ms_e2e, tokens_per_sec_e2e
# avg_step_ms_compute, tokens_per_sec_compute
# peak_mem_alloc_gb, peak_mem_reserved_gb
# epoch_wall_s
```

---

## Reproducibility

```python
from utils import set_seed
set_seed(42)   # seeds Python, NumPy, PyTorch CPU+GPU; enables cuDNN deterministic mode
```

---

## Citation

If you use this code, please cite our paper:

```bibtex
@article{amiraslani2026homa,
  title   = {HOMA: Higher-Order Modular Attention for Protein Sequence Modelling},
  author  = {Amiraslani, Shirin and others},
  journal = {arXiv preprint arXiv:2603.11133},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.11133},
}
```

---

## License

Copyright (c) 2025 Shirin Amiraslani
All Rights Reserved.

You may not copy, modify, distribute, or use this software for
any commercial or research purposes without explicit written
permission from the author.
