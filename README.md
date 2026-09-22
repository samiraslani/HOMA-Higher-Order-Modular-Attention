# HOMA — Higher-Order Modular Attention

HOMA (Higher-Order Modular Attention) is a sequence transformer that extends standard pairwise attention with third-order (triadic) interactions between sequence positions. The core contribution is a unified architecture that **fuses pairwise attention with triadic attention** through a learned MLP — capturing richer positional dependencies while remaining computationally tractable.

Standard transformers compute pairwise interactions via scaled dot-product attention:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_h}}\right)V$$

HOMA introduces a **third-order interaction term** through an additional low-rank projection $U$:

$$\text{scores}_{3D}[i,j,k] = \frac{(Q_i \odot K_j^{(w)} \odot U_k^{(w)}) \cdot \mathbf{1}}{\sqrt{d_h}}, \quad j,k \in [0,w)$$

where $j$ and $k$ index positions within a local window of size $w$ around query position $i$. The triadic attention reads out the value product $V_j \odot V_k$. The 2D and 3D outputs are fused by a small MLP, giving the model access to both pairwise and triadic structure simultaneously. Three design choices keep computation tractable:

| # | Technique | Effect |
|---|---|---|
| 1 | **Overlapping blocks** | The sequence of length $L$ is partitioned into $T$ overlapping blocks of length $\ell$, applied to both the pairwise and triadic branches. This reduces global $O(L^2)$ / $O(L^3)$ cost to a block-local computation. |
| 2 | **Local window for triadic attention** | Within each block, the triadic branch further restricts interactions to a sliding window of size $w$ around each query position, reducing the per-block triadic cost from $O(\ell^3)$ to $O(\ell \cdot w^2)$. |
| 3 | **Low-rank U-matrix** | The third projection is factorised as $U = W_{u_u} W_{u_v}$ with inner rank $r$, cutting 3D parameters by ~97% (262 k → 8 k for $d$=512, $r$=8). |

Transfer learning is supported for training the HOMA attention mechanism: pretrained 2D weights ($W_q, W_k, W_v$) can be loaded from a `blockwise2d` checkpoint and optionally frozen, so only the 3D-specific parameters ($W_{u_u}$, $W_{u_v}$, fusion MLP) are trained from scratch.

The code covers the diagnostic tasks (PARITY / MAJORITY, MATCH2 / MATCH3) and the [TAPE](https://github.com/songlab-cal/tape) protein tasks (secondary structure, contact prediction, fluorescence) described in the paper.

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
- [Running the experiments](#running-the-experiments)
- [Training output](#training-output)
- [Best-model selection](#best-model-selection)
- [Checkpointing](#checkpointing)
- [Efficiency tracking](#efficiency-tracking)
- [Citation](#citation)
- [License](#license)

---

## Package structure

```
HOMA-Higher-Order-Modular-Attention-new
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml                    # package metadata (optional install)
├── config.py                         # ModelConfig, AttentionConfig, TrainingConfig
├── models/
│   ├── attention/
│   │   ├── base.py                   # AttentionBase, shared sliding-block helpers
│   │   ├── attention_2d.py           # MultiHeadAttn2D, Attn2DBlockwise, Attn2DLinformer
│   │   ├── attention_3d.py           # HOMA (main contribution), MultiHeadAttn3D
│   │   └── __init__.py               # get_attention() factory
│   ├── feedforward.py                # FeedForward
│   ├── encoder.py                    # Encoder layer
│   └── protein_transformer.py        # ProteinTransformer, PerResidueHead, GlobalRegressionHead
├── tasks/
│   ├── diagnostic/                   # synthetic tasks; numpy + torch only
│   │   ├── mechanisms.py             # registry of every attention arm, by name
│   │   ├── parity_majority.py        # PARITY-k / MAJORITY-k data, TinyModel, run_one
│   │   └── match.py                  # MATCH2 / MATCH3 data, Fourier embedding, run_one
│   └── protein/                      # TAPE protein-sequence tasks
│       ├── secondary_structure.py    # SecondaryStructureTask
│       ├── contact_prediction.py     # ContactPredictionTask, ContactDataset, P@L/k metrics
│       └── fluorescence.py           # FluorescenceTask
├── data/
│   ├── datasets.py                   # SecondaryStructureDataset, FluorescenceDataset
│   ├── collate.py                    # collate_ss3, collate_regression
│   └── tape_compat.py                # TAPETokenizer, LMDBDataset (replaces tape_proteins)
├── training/
│   ├── trainer.py                    # Trainer (unified loop for the protein tasks)
│   ├── trajectory.py                 # TrajectoryTrainer (per-epoch test curves)
│   └── efficiency.py                 # EfficiencyTracker (timing + memory)
├── evaluation/
│   └── metrics.py                    # accuracy_per_position, spearman_correlation
├── utils/
│   ├── seed.py                       # set_seed()
│   └── checkpointing.py              # save_checkpoint, load_checkpoint
└── experiments/                      # one runner per experiment
    ├── common.py                     # resumable result store, device selection, shared CLI
    ├── run_parity.py                 # PARITY-k / MAJORITY-k across order, depth and width
    ├── run_match.py                  # MATCH2 / MATCH3 across sequence length and width
    ├── run_coverage.py               # triadic window coverage and depth
    └── run_tape.py                   # secondary structure, contact prediction, fluorescence
```

---

## Installation

Requirements:
- Python >= 3.9
- PyTorch >= 2.1

```bash
git clone https://github.com/samiraslani/HOMA-Higher-Order-Modular-Attention.git
cd HOMA-Higher-Order-Modular-Attention
pip install -r requirements.txt     # torch, numpy, scipy, lmdb, tqdm
```

Run everything from the repository root; the scripts in `experiments/` find the code folders next to them, so installing the code itself is not required. `pip install -e .` also works and makes the modules importable from anywhere, but it installs them under their folder names (`models`, `data`, `utils`, …), which can clash with other installed packages, so use a separate virtual environment if you install.

The diagnostic tasks need only `torch` and `numpy`; the protein tasks also need `scipy` and `lmdb`.

`tape_proteins` is **not** needed. It no longer imports under NumPy 2, so the two pieces of it this code uses — the IUPAC tokenizer and the LMDB reader — are reproduced in `data/tape_compat.py` with identical behaviour.

---

## Dataset setup

The **diagnostic tasks generate their data** from a seed; nothing needs to be downloaded.

The **protein tasks** use the TAPE LMDB files, which this repository does not include. Download them following the [TAPE instructions](https://github.com/songlab-cal/tape#datasets) and place them under one folder:

```
DATA_ROOT/
  secondary_structure/
    secondary_structure_train.lmdb
    secondary_structure_valid.lmdb
    secondary_structure_cb513.lmdb
    secondary_structure_casp12.lmdb
    secondary_structure_ts115.lmdb
  fluorescence/
    fluorescence_train.lmdb
    fluorescence_valid.lmdb
    fluorescence_test.lmdb
  proteinnet/
    proteinnet_train.lmdb
    proteinnet_valid.lmdb
    proteinnet_test.lmdb
```

File names may also be `<split>.lmdb` (e.g. `train.lmdb`). Pass the folder as `--data-root` (or set `$TAPE_DATA_DIR`). ProteinNet can live elsewhere: pass `--proteinnet DIR` or set `$PROTEINNET_DIR`.

The TAPE download mirror currently returns HTTP 403, so the data has to come from an existing copy.

**Note**: the runners raise a `FileNotFoundError` if the expected LMDB files are not present.

---

## Architecture

<img width="744" height="396" alt="HOMA architecture" src="https://github.com/user-attachments/assets/b046c42a-00a9-4284-97e6-147a3634eab8" />

HOMA combines a blockwise pairwise (2D) attention pathway with a parallel windowed triadic (3D) attention pathway. Starting from shared linear projections $Q$, $K$, $U$ and $V$, both branches compute attention within overlapping local blocks. Their outputs are concatenated and passed through a fusion MLP to produce the final per-position representation. Greyed squares indicate the local overlapping regions attended by each attention mechanism.

---

## Attention mechanisms

Five attention types are available via `get_attention(type, ...)` or `AttentionConfig(type=...)`:

| Type | Class | Description |
|---|---|---|
| `"plain2d"` | `MultiHeadAttn2D` | Standard scaled dot-product attention (Vaswani et al., 2017) |
| `"blockwise2d"` | `Attn2DBlockwise` | Pairwise attention over overlapping blocks |
| `"blockwise3d"` | `MultiHeadAttn3D` | Windowed triadic block attention only, no 2D branch |
| `"homa"` | `HOMA` | **Main contribution** — fusion of blockwise pairwise and windowed triadic block attention |

`HOMA` also takes `combine="add"` (a plain sum of the two branches instead of the fusion MLP, called HOMA-add) and `combine="gated"` (a sum with one learnable scalar weight on the triadic branch).

### Complexity comparison

| Variant | Compute | Memory |
|---|---|---|
| `plain2d` — Standard 2D | $O(L^2 d_h)$ | $O(L^2)$ |
| `blockwise2d` — Blockwise 2D | $O(T\,\ell^2 d_h)$ | $O(T\,\ell^2)$ |
| `blockwise3d` — Blockwise triadic | $O(T\,\ell\,w^2 d_h)$ | $O(T\,\ell\,w^2)$ |
| `homa` — Pairwise + triadic fusion | $O\left(T\,\ell^2 d_h + T\,\ell\,w^2 d_h\right)$ | $O\left(T\,\ell^2 + T\,\ell\,w^2 d_h\right)$ |

Per-block, per-head HOMA cost:

$$\text{Compute}_{\text{HOMA}} = \mathcal{O}\left(\ell^{2} d_{\text{head}} + \ell\, w^{2} d_{\text{head}}\right)$$

$$\text{Memory}_{\text{HOMA}} = \mathcal{O}\left(\ell^{2} + \ell\, w^{2} d_{\text{head}}\right)$$

$T$ = number of overlapping blocks, $\ell$ = block size (default 30), $w$ = window size (default 7), $k$ = Linformer projection dimension, $d_h$ = head dimension, $L$ = sequence length.

---

## Quick start

### Diagnostic tasks

```python
from tasks.diagnostic import run_one, DEFAULT_CFG, match_run_one, PUBLISHED_M

rec = run_one("homa", family="parity", k=4, d_model=64, seed=0,
              n_layers=1, cfg=DEFAULT_CFG, device="cuda")
print(rec["final"], rec["params"])      # final test accuracy, parameter count

rec = match_run_one("homa", order=3, N=6, M=PUBLISHED_M[(3, 6)], d_model=32,
                    heads=4, seed=0, device="cuda",
                    epochs=40, train_n=30000, test_n=2000, lr=3e-3, rank=8)
print(rec["final"], rec["majority"])    # final test accuracy, majority-class baseline
```

### Secondary structure prediction (SS3)

```python
from data.tape_compat import LMDBDataset, TAPETokenizer
from config import ModelConfig, AttentionConfig, TrainingConfig
from tasks.protein.secondary_structure import SecondaryStructureTask

model_cfg = ModelConfig(d_model=512, num_layers=12, num_heads=8, max_seq_length=512)
attn_cfg  = AttentionConfig(type="homa", block_size=30, stride=15, window_size=3)
train_cfg = TrainingConfig(batch_size=16, learning_rate=1e-4, epochs=10)

tokenizer = TAPETokenizer(vocab="iupac")
task = SecondaryStructureTask(model_cfg, attn_cfg, train_cfg)

model, history = task.train(
    train_lmdb=LMDBDataset("DATA_ROOT/secondary_structure/secondary_structure_train.lmdb"),
    val_lmdb=LMDBDataset("DATA_ROOT/secondary_structure/secondary_structure_valid.lmdb"),
    tokenizer=tokenizer,
    track_efficiency=True,
)
```

Set `max_seq_length` when training through `task.train()`: with the default `None`, batches are padded to different lengths and the trainer cannot concatenate the per-batch predictions to compute the epoch metric.


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
import torch
from config import ModelConfig, AttentionConfig
from models import ProteinTransformer, PerResidueHead

model_cfg = ModelConfig()
attn_cfg  = AttentionConfig(type="homa", block_size=30, stride=15)
head      = PerResidueHead(d_model=512, num_classes=3)
model     = ProteinTransformer(model_cfg, attn_cfg, head)

x = torch.randint(1, 30, (2, 128))        # (batch=2, length=128)
logits, _ = model(x, labels=torch.zeros(2, 128, dtype=torch.long))
print(logits.shape)                        # (2, 150, 3): padded to a length the 30/15 blocks tile
```

---

## Configuration

All hyperparameters of the protein models are defined as Python dataclasses in `config.py`:
- **`ModelConfig`** — vocabulary, width, depth, heads, FFN width, dropout, maximum length
- **`AttentionConfig`** — attention type, block size and stride, triadic window, rank of $U$, the 2D/3D combination, transfer learning
- **`TrainingConfig`** — batch size, learning rate, epochs, warm-up and schedule, gradient clipping, checkpoint directory

See `config.py` for full defaults and parameter documentation.

The diagnostic tasks take a small dictionary instead (`DEFAULT_CFG` in `tasks/diagnostic/parity_majority.py`; keyword arguments of `run_one` in `tasks/diagnostic/match.py`).

---

## Running the experiments

```bash
python experiments/run_parity.py                       # PARITY / MAJORITY, every order, depth and width
python experiments/run_parity.py --preset capacity     # PARITY-3, one layer, widths 8-128
python experiments/run_parity.py --preset long         # Pairwise-2D on PARITY-5, 120 epochs
python experiments/run_match.py                        # MATCH3
python experiments/run_match.py --orders 2             # MATCH2
python experiments/run_coverage.py                     # triadic window coverage and depth
python experiments/run_tape.py --data-root DATA_ROOT   # secondary structure, contact, fluorescence
python experiments/run_tape.py --data-root DATA_ROOT --tasks contact --widths 64
```

Results are written to `results/` after every run, and a restarted runner skips what is already done. Every runner takes `--quick` for a short pipeline check, `--tables-only` to print a summary of an existing results file, and `--help` for all options.

---

## Training output

At the start of every protein training run the trainer prints a one-time setup summary so you can verify the configuration at a glance. The examples below are the real output of a small run (2 layers, $d_{\mathrm{model}}=32$, CPU).

**Standard run:**

```
--------------------------------------------------
  Training Setup
--------------------------------------------------
  Attention type    : homa
  Rank              : 8
  Window size       : 3
  2D/3D combine     : fusion — concat -> MLP(2*head_dim -> 128 -> head_dim)
  Device            : cpu
  Epochs            : 1
  Learning rate     : 0.0001
  LR scheduler      : none
  Grad clip         : disabled
  U-entropy penalty : disabled
  Best-model by     : val_loss
  Trainable params  : 163,755
--------------------------------------------------
```

**With transfer learning** (`pretrained_ckpt` is set), the model first reports which checkpoint was used to initialize the 2D projection weights:

```
  Transfer learning : blockwise 2D parameters (W_q, W_k, W_v) loaded from: checkpoints/blockwise2d.pt
```

**With frozen layers** (`freeze_2d=True`), the summary additionally shows which modules are frozen and how many parameters are excluded from the optimiser:

```
  Transfer learning : blockwise 2D parameters (W_q, W_k, W_v) loaded from: checkpoints/blockwise2d.pt
  Frozen layers     : W_q, W_k, W_v  (only W_u_u, W_u_v, fusion_layer will be trained)
--------------------------------------------------
  Training Setup
--------------------------------------------------
  ...
  Trainable params  : 157,419
  Frozen params     : 6,336
  Frozen modules    : encoder_layers.0.mha.W_k, encoder_layers.0.mha.W_q, encoder_layers.0.mha.W_v, ...
--------------------------------------------------
```

Each epoch then prints a one-line progress update, with the best epoch flagged, and the best weights are restored at the end:

```
Epoch 1/1 | Train Loss: 1.2458 | Train metric: 0.3391 | Val Loss: 1.1966 | Val metric: 0.3547  ← best
Loaded best model from epoch 1 (val_loss=1.1966, val_metric=0.3547)
```

With `track_efficiency=True`, each epoch also prints its timing and memory (on CUDA; see [Efficiency tracking](#efficiency-tracking)).

The diagnostic runners print one line per run instead, with the final accuracy and an estimate of the time remaining.

---

## Best-model selection

For the protein tasks the reported test score comes from the best-validation epoch, chosen by a task-specific criterion:

`Trainer` (used by `task.train()`) writes two checkpoint files per run:

| File | Contents |
|---|---|
| `{attn_name}.pt` | Rolling last-epoch checkpoint (used for resuming interrupted runs). |
| `{attn_name}_best.pt` | Snapshot of the best epoch according to `select_by`. |

At the end of `Trainer.fit()` the best weights are loaded back into the model before it is returned, so downstream evaluation always uses the best checkpoint — not the final epoch.

```python
# select_by is set automatically by each task wrapper, but can be overridden:
from training.trainer import Trainer

trainer = Trainer(config=train_cfg, attn_name="homa", select_by="val_loss")   # SS3
trainer = Trainer(config=train_cfg, attn_name="homa", select_by="val_metric") # fluorescence
```

`run_tape.py` uses `TrajectoryTrainer`, which additionally evaluates every test split at every epoch (never using it for selection) and reads the test score at the best-validation epoch.

The diagnostic tasks report the accuracy after the last epoch.

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

The experiment runners resume at the level of whole runs instead: each finished run is saved to the results file immediately, and a restarted runner skips it.

---

## Efficiency tracking

Pass `track_efficiency=True` to any protein task's `train()` method to collect per-step timing and memory statistics:

```python
model, history = task.train(..., track_efficiency=True)

# history keys include:
# avg_step_ms_e2e, tokens_per_sec_e2e
# avg_step_ms_compute, tokens_per_sec_compute
# peak_mem_alloc_gb, peak_mem_reserved_gb
# epoch_wall_s
```

The per-step timing and memory statistics are recorded on CUDA only; on CPU or Apple MPS they are reported as `nan`, while `epoch_wall_s` is always recorded. The diagnostic runners record wall time and, on CUDA, peak memory for every run.

---

## Citation

If you use this code, please cite our paper:

```bibtex
@article{amiraslani2026homa,
  title   = {HOMA: Higher-Order Modular Attention for Protein Sequence Modelling},
  author  = {Amiraslani, Shirin and Gao, Xin},
  journal = {arXiv preprint arXiv:2603.11133},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.11133},
}
```

---

## License

This code is released under the **Apache License 2.0**. See `LICENSE`.
