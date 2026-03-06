# TAPE BioTransformer

A protein sequence transformer with **sliding-window 3D attention** that captures higher-order (three-way) interactions between sequence positions.  Built on the [TAPE benchmark](https://github.com/songlab-cal/tape) and evaluated on secondary structure prediction, fluorescence prediction, and stability prediction.

---

## Overview

Standard transformers compute pairwise interactions between sequence positions via scaled dot-product attention:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_h}}\right)V$$

This work introduces a **third-order interaction term** through an additional learnable projection $L$:

$$\text{scores}_{3D}[i,j,k] = \frac{(Q_i \odot K_j^{(w)} \odot L_k^{(w)}) \cdot \mathbf{1}}{\sqrt{d_h}}, \quad j,k \in [0,w)$$

where $j$ and $k$ index positions within a local window of size $w$ around query position $i$.  The 2D and 3D outputs are fused by a small MLP.

Two design choices keep the computation tractable:

| Technique | Effect |
|---|---|
| **Sliding-window blocking** | Restricts attention to blocks of length `block_size`, reducing complexity from $O(L^2)$ / $O(L^3)$ to $O(L \cdot b^2)$ / $O(L \cdot w^2)$ |
| **Low-rank L-matrix** | Factorises $L = x W_{l_u} W_{l_v}$ with rank 8, cutting 3D parameters by ~97% (262 k → 8 k for d=512) |

Transfer learning is supported: pretrained 2D weights ($W_q, W_k, W_v$) can be loaded from a strong 2D baseline and optionally frozen, so only the 3D components ($W_{l_u}, W_{l_v}$, fusion MLP) are trained from scratch.

---

## Package structure

```
tape_biotransformer/
├── config.py                       # ModelConfig, AttentionConfig, TrainingConfig
├── data/
│   ├── datasets.py                 # SecondaryStructureDataset, FluorescenceDataset, StabilityDataset
│   └── collate.py                  # collate_ss3, collate_regression
├── models/
│   ├── attention/
│   │   ├── base.py                 # AttentionBase, shared sliding-block helpers
│   │   ├── attention_2d.py         # MultiHeadAttn2D, Attn2D_MultiPed, Attn2DLinformer
│   │   ├── attention_3d.py         # Attn2DMultiPed3Dslidingw  ← main contribution
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

**Requirements:** Python ≥ 3.8, PyTorch ≥ 1.13

```bash
git clone https://github.com/your-org/tape-biotransformer.git
cd tape-biotransformer
pip install -e .
```

TAPE datasets and tokenizer are provided by the `tape_proteins` package:

```bash
pip install tape_proteins
```

Download the benchmark data following the [TAPE instructions](https://github.com/songlab-cal/tape#datasets).

---

## Quick start

### Secondary structure prediction (SS3)

```python
from tape.datasets import LMDBDataset
from tape.tokenizers import TAPETokenizer

from tape_biotransformer.config import ModelConfig, AttentionConfig, TrainingConfig
from tape_biotransformer.tasks.secondary_structure import SecondaryStructureTask

model_cfg = ModelConfig(d_model=512, num_layers=12, num_heads=8)
attn_cfg  = AttentionConfig(type="sliding3d", block_size=40, stride=15, window_size=7)
train_cfg = TrainingConfig(batch_size=16, learning_rate=1e-4, epochs=20)

tokenizer = TAPETokenizer(vocab="iupac")
task = SecondaryStructureTask(model_cfg, attn_cfg, train_cfg)

model, history = task.train(
    train_lmdb=LMDBDataset("data/secondary_structure/train.lmdb"),
    val_lmdb=LMDBDataset("data/secondary_structure/valid.lmdb"),
    tokenizer=tokenizer,
    track_efficiency=True,
)
```

### Fluorescence / stability prediction

```python
from tape_biotransformer.tasks.fluorescence import FluorescenceTask

task = FluorescenceTask(model_cfg, attn_cfg, train_cfg)
model, history = task.train(train_lmdb, val_lmdb, tokenizer)
```

### Transfer learning from a 2D baseline

```python
attn_cfg = AttentionConfig(
    type="sliding3d",
    block_size=40,
    stride=15,
    window_size=7,
    rank_3d=8,
    pretrained_ckpt="checkpoints-ss3/multiped2d.pt",  # pretrained 2D checkpoint
    freeze_2d=False,   # set True to freeze W_q/W_k/W_v
)
```

### Using the model directly

```python
from tape_biotransformer.models import ProteinTransformer, PerResidueHead
from tape_biotransformer.config import ModelConfig, AttentionConfig

model_cfg = ModelConfig()
attn_cfg  = AttentionConfig(type="sliding3d", block_size=40, stride=15)
head      = PerResidueHead(d_model=512, num_classes=3)
model     = ProteinTransformer(model_cfg, attn_cfg, head)

import torch
x = torch.randint(1, 30, (2, 128))        # (batch=2, length=128)
logits, _ = model(x, labels=torch.zeros(2, 128, dtype=torch.long))
print(logits.shape)                        # (2, 128, 3)
```

---

## Attention mechanisms

Four attention types are available via `get_attention(type, ...)` or `AttentionConfig(type=...)`:

| Type | Class | Description |
|---|---|---|
| `"plain2d"` | `MultiHeadAttn2D` | Standard $O(L^2)$ scaled dot-product attention |
| `"multiped2d"` | `Attn2D_MultiPed` | Sliding-window 2D attention — $O(L \cdot b^2)$ |
| `"linformer2d"` | `Attn2DLinformer` | Low-rank 2D attention — $O(L \cdot k)$ |
| `"sliding3d"` | `Attn2DMultiPed3Dslidingw` | **Main contribution** — sliding-window 3D with low-rank $L$ |

### Complexity comparison

| Variant | Attention complexity | Memory |
|---|---|---|
| Standard 2D | $O(L^2)$ | $O(L^2)$ |
| Sliding-window 2D | $O(L \cdot b^2)$ | $O(L \cdot b)$ |
| Standard 3D | $O(L^3)$ | $O(L^2)$ |
| **Sliding-window 3D** | $O(L \cdot w^2)$ | $O(L \cdot w)$ |

$b$ = block size (default 40), $w$ = window size (default 7), $L$ = sequence length.

---

## Configuration

All hyperparameters are defined in `config.py` as Python dataclasses with sensible defaults:

```python
from tape_biotransformer.config import ModelConfig, AttentionConfig, TrainingConfig

# Architecture
model_cfg = ModelConfig(
    vocab_size=30,           # IUPAC amino-acid vocabulary
    d_model=512,
    num_layers=12,
    num_heads=8,
    dim_feedforward=1024,
    dropout=0.4,
    max_seq_length=512,
)

# Attention
attn_cfg = AttentionConfig(
    type="sliding3d",
    block_size=40,       # tokens per sliding block
    stride=15,           # step between blocks
    window_size=7,       # local 3D context window
    rank_3d=8,           # L-matrix inner rank
    pretrained_ckpt=None,
    freeze_2d=False,
)

# Training
train_cfg = TrainingConfig(
    batch_size=16,
    learning_rate=1e-4,
    epochs=20,
    checkpoint_dir="checkpoints",
)
```

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

Each script trains three model variants sequentially:
1. `plain2d` — standard 2D baseline
2. `multiped2d` — sliding-window 2D baseline
3. `sliding3d` — our 3D contribution (optionally initialised from the `multiped2d` checkpoint)

---

## Checkpointing

Checkpoints are saved automatically after every epoch and support seamless resumption:

```python
# Training resumes from the last checkpoint automatically
trainer = Trainer(config=train_cfg, attn_name="sliding3d")
model, history = trainer.fit(model, train_loader, val_loader, ...)
```

Manual save / load:

```python
from tape_biotransformer.utils import save_checkpoint, load_checkpoint

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
from tape_biotransformer.utils import set_seed
set_seed(42)   # seeds Python, NumPy, PyTorch CPU+GPU; enables cuDNN deterministic mode
```

---

## Citation

If you use this code, please cite our paper:

```bibtex
@article{biotransformer3d,
  title   = {BioTransformer with Sliding-Window 3D Attention for Protein Sequence Modelling},
  author  = {[Authors]},
  journal = {[Journal / Conference]},
  year    = {[Year]},
}
```

---

## License
Copyright (c) 2025 Shirin Amiraslani  
All Rights Reserved.

You may not copy, modify, distribute, or use this software for
any commercial or research purposes without explicit written
permission from the author.
