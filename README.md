# HOMA: Higher-Order Modular Attention

HOMA is a self-attention layer that adds a *triadic* term to ordinary pairwise
attention. Each query scores pairs of positions with a trilinear form over
learned projections `Q`, `K` and a low-rank `U`, and reads out the value product
`V_j ⊙ V_k`. The pairwise and triadic outputs are computed in parallel from
shared projections and fused.

This repository holds the attention modules and the code for the paper's
experiments: the diagnostic tasks (PARITY / MAJORITY and MATCH2 / MATCH3) and
the three TAPE protein tasks (secondary structure, contact prediction and
fluorescence). Each experiment is one script, and each script writes its
results in a resumable JSON file.

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
│   │   ├── attention_2d.py           # Pairwise-2D, Blockwise-2D, Linformer
│   │   ├── attention_3d.py           # HOMA (main contribution), Blockwise-3D
│   │   └── __init__.py               # get_attention() factory
│   ├── feedforward.py                # FeedForward
│   ├── encoder.py                    # Encoder layer
│   └── protein_transformer.py        # ProteinTransformer, PerResidueHead, GlobalRegressionHead
├── tasks/
│   ├── diagnostic/                   # controlled tasks; numpy + torch only
│   │   ├── mechanisms.py             # registry of every attention arm, by its paper name
│   │   ├── parity_majority.py        # PARITY-k / MAJORITY-k data, TinyModel, run_one
│   │   └── match.py                  # MATCH2 / MATCH3 data, Fourier embedding, run_one
│   └── protein/                      # TAPE protein-sequence tasks
│       ├── secondary_structure.py    # SecondaryStructureTask (Q3)
│       ├── contact_prediction.py     # ContactPredictionTask, ProteinNet data, P@L/5
│       └── fluorescence.py           # FluorescenceTask (Spearman rho)
├── data/
│   ├── datasets.py                   # secondary-structure and fluorescence datasets
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
    ├── run_parity.py                 # PARITY / MAJORITY: Table 1, Figure 2
    ├── run_match.py                  # MATCH2 / MATCH3: Table 2
    ├── run_coverage.py               # window coverage and depth: Figure 4
    └── run_tape.py                   # secondary structure, contact, fluorescence: Figure 3
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # torch, numpy, scipy, lmdb, tqdm
```

Run everything from the repository root. The runners in `experiments/` find the
code folders next to them, so no installation of the code itself is needed.
`pip install -e .` also works and makes the modules importable from anywhere,
but it installs them under their folder names (`models`, `data`, `utils`, …),
which can clash with other installed packages, so a separate virtual
environment is recommended.

The diagnostic tasks need only `torch` and `numpy`; the protein tasks also need
`scipy` and `lmdb`.

`tape_proteins` is **not** needed. It no longer imports under NumPy 2, so the two
pieces of it this code used, the IUPAC tokenizer and the LMDB reader, are
reproduced in `data/tape_compat.py` with identical behaviour.

Tested with Python 3.12 under torch 2.7.1 and torch 2.14 (NumPy 2.2 and 2.5).

## What produces what

| Paper artifact | Command | Cost |
|---|---|---|
| **Table 1**: PARITY-k by order, depth and width, with the MAJORITY control and lower orders; **Figure 2** | `python experiments/run_parity.py` | 720 runs; ~30 s each at depth 1 on a laptop GPU, more with depth |
| PARITY-3 across widths | `python experiments/run_parity.py --preset capacity` | 60 runs |
| Pairwise-2D at 120 epochs | `python experiments/run_parity.py --preset long` | 9 runs |
| **Table 2**: MATCH3 by length and width | `python experiments/run_match.py` | 144 runs; ~30 s each on a T4, ~5 min on a laptop GPU |
| MATCH2 control | `python experiments/run_match.py --orders 2` | |
| **Figure 4**: window coverage and depth | `python experiments/run_coverage.py` | 225 runs, ~10 s each on CPU |
| **Figure 3**: secondary structure, contact prediction, fluorescence | `python experiments/run_tape.py --data-root DIR` | GPU; hours per width, ~21 A100-hours for contact |

Every runner:

* writes its results file after **each** run, and on restart skips what it has
  already done, so an interrupted session loses at most one run;
* refuses to append to a results file written under a different protocol;
* takes `--quick` for a short end-to-end check (a small budget; the numbers are
  not the published ones), and `--tables-only` to print the tables from an
  existing file;
* takes `--seeds`, `--device`, and axis flags (`--widths`, `--orders`,
  `--tasks`, …) to run part of a grid.

`run_tape.py` runs all three protein tasks by default; `--tasks` selects a
subset, for example `--tasks contact`. It writes `results/tape.json`
(secondary structure and fluorescence) and `results/contact.json`.

### How close a rerun gets

Seeding makes a run repeatable on one machine with one torch version. It does
not make runs identical across machines. On these tasks a model often sits
right at the edge of solving the problem, so small numerical differences can
grow:

* **CPU, torch 2.7.1**: re-running published Table 3 and window-coverage cells
  reproduces their recorded accuracy exactly, to every printed digit.
* **CPU, torch 2.14**: the same cells agree to within a few of 8,000 test labels.
* **Apple MPS**: re-running Table 1 at d=64, depth 1, 9 of 12 PARITY cells land
  within 0.02 of the published mean. The other 3 move by up to 0.07: at
  PARITY-5, Blockwise-3D and HOMA-add land at 0.886 and 0.882 against a
  published 0.938 and 0.948, which swaps their order. Both stay far above
  Pairwise-2D and below HOMA.

Compare seed means, not single runs.

## Protocol

These are the settings that produced the published numbers. The runners
hard-code them as `PUBLISHED_CFG`, `TASK_SPEC` and `CONTACT`.

| | PARITY / MAJORITY | MATCH | Sec. Structure | Contact | Fluorescence |
|---|---|---|---|---|---|
| train / test | 3,000 / 800 | 30,000 / 2,000 | TAPE splits | ProteinNet | TAPE splits |
| length | 16 | N = 6, 8 | 512 | crop 256 | 237 |
| width | 32, 64 | 8–256 | 32–512 | 32–256 | 32–256 |
| layers | 1, 6, 12 | 1 | 12 | 12 | 12 |
| heads | 4 | 4 | 8 | 8 | 8 |
| FFN | none | none | 2d | 2d | 2d |
| dropout | 0 | 0 | 0.4 | 0.1 | 0.1 |
| block / stride | one block | one block | 30 / 15 | 32 / 16 | 30 / 15 |
| triadic window | 7 | 2N−1 | 5 | 5 | 5 |
| U rank | 8 | 8 | 8 | 8 | 8 |
| optimizer | Adam 2e-3 | Adam 3e-3 | Adam 1e-4 | AdamW 1e-4, wd 0.01 | Adam 1e-4 |
| schedule, clipping | none | none | none | OneCycle 10 %, clip 1.0 | none |
| batch, epochs | 128, 40 | 256, 40 | 16, 10 | 8, 8 | 16, 10 |
| reported | last epoch | last epoch | test at best val. loss | test after last epoch | test at best val. ρ |
| seeds | 0, 1, 2 | 0, 1, 2 | 42, 456, 608 | 0, 1, 2 | 42, 456, 608 |

Details that are easy to get wrong:

* **No FFN in the diagnostic models.** A feed-forward sublayer can compute
  parity by itself and would hide what the attention contributes.
* **Residual connections.** Table 1 builds models with residual connections at
  every depth, including depth 1. The PARITY half of Table 3 uses a single
  attention layer with no residual path. Their one-layer cells are therefore
  different models.
* **Pairwise-2D in the diagnostic tables** is run as `blockwise2d` with one block
  spanning the whole sequence, which is the same operator.
* **MATCH modulus.** `M` is pinned per (order, N) in `PUBLISHED_M`, and the
  runner checks at startup that the calibration still reproduces it. The
  embedding is a frozen Fourier basis with a learned projection on top.
* **Fluorescence length is 237**, the longest GFP sequence. The regression head
  reads `max_len × d_model` features, so this setting changes the model, not
  just the padding. The published parameter counts match 237 at every width and
  arm, and none match 512.

## Data

All protein data sits under one folder, passed as `--data-root` (or
`$TAPE_DATA_DIR`):

```
DATA_ROOT/
├── secondary_structure/   secondary_structure_{train,valid,cb513,casp12,ts115}.lmdb
├── fluorescence/          fluorescence_{train,valid,test}.lmdb
└── proteinnet/            proteinnet_{train,valid,test}.lmdb
```

File names may also be `<split>.lmdb`. ProteinNet can live elsewhere: pass
`--proteinnet DIR` or set `$PROTEINNET_DIR`. The contact loader checks that
coordinates are in Ångströms before thresholding at 8 Å; an 8.0 threshold on
picometre coordinates would fail silently, not loudly.

The TAPE download mirror currently returns HTTP 403, so the data has to come
from an existing copy.

## Licence

Apache 2.0. See `LICENSE`.
