# HOMA: Higher-Order Modular Attention

HOMA is a self-attention layer that adds a *triadic* term to ordinary pairwise
attention. Each query scores pairs of positions with a trilinear form over
learned projections `Q`, `K` and a low-rank `U`, and reads out the value product
`V_j ⊙ V_k`. The pairwise and triadic outputs are computed in parallel from
shared projections and fused.

This repository holds the attention modules and **the code that produces every
number in the paper and its supplement**: the tables, and the results the
figures plot. The plotting scripts themselves are not included. Each experiment is one
script, and each script writes its results in the same format as the original
result files. You can check a rerun against the published numbers cell by cell.

```
HOMA-Higher-Order-Modular-Attention-new
├── README.md
├── LICENSE
├── pyproject.toml                        # package metadata; extras: tape, figures, test, all
├── requirements.txt
├── Makefile                              # one target per experiment, plus test / quick / verify
├── homa/                                 # the library
│   ├── config.py                         # ModelConfig, AttentionConfig, TrainingConfig
│   ├── models/
│   │   ├── attention/
│   │   │   ├── base.py                   # AttentionBase, shared sliding-block helpers
│   │   │   ├── attention_2d.py           # Pairwise-2D, Blockwise-2D, Linformer
│   │   │   ├── attention_3d.py           # HOMA (main contribution), Blockwise-3D,
│   │   │   │                             #   dense triadic attention
│   │   │   ├── linear_value.py           # linear-value ablation (Table 3)
│   │   │   └── __init__.py               # get_attention() factory
│   │   ├── feedforward.py                # FeedForward
│   │   ├── encoder.py                    # Encoder layer
│   │   └── protein_transformer.py        # ProteinTransformer, PerResidueHead, GlobalRegressionHead
│   ├── synthetic/                        # controlled tasks; numpy + torch only
│   │   ├── mechanisms.py                 # registry of every attention arm, by its paper name
│   │   ├── order_tasks.py                # PARITY-k / MAJORITY-k data, TinyModel, run_one
│   │   ├── match_tasks.py                # MATCH-q data, Fourier embedding, PUBLISHED_M, run_one
│   │   └── __init__.py                   # public API of both task families
│   ├── contact/                          # contact prediction on ProteinNet
│   │   ├── data.py                       # ContactDataset, unit check, collate_contacts
│   │   ├── model.py                      # ContactModel, PairwiseContactHead, build_contact_model
│   │   ├── metrics.py                    # precision at L/k by separation range
│   │   └── __init__.py
│   ├── data/
│   │   ├── datasets.py                   # TAPE dataset wrappers
│   │   ├── collate.py                    # collate_ss3, collate_regression
│   │   └── tape_compat.py                # TAPETokenizer, LMDBDataset (replaces tape_proteins)
│   ├── tasks/
│   │   ├── secondary_structure.py        # SecondaryStructureTask
│   │   ├── fluorescence.py               # FluorescenceTask
│   │   └── stability.py                  # StabilityTask (not in the paper)
│   ├── training/
│   │   ├── trainer.py                    # Trainer (unified loop for all tasks)
│   │   ├── trajectory.py                 # TrajectoryTrainer (per-epoch test curves)
│   │   └── efficiency.py                 # EfficiencyTracker (timing + memory)
│   ├── evaluation/
│   │   └── metrics.py                    # accuracy_per_position, spearman_correlation
│   └── utils/
│       ├── seed.py                       # set_seed()
│       └── checkpointing.py              # save_checkpoint, load_checkpoint
├── experiments/                          # one runner per experiment
│   ├── common.py                         # resumable result store, device selection, shared CLI
│   ├── run_parity.py                     # Table 1, Tables S4-S5, Figure 2
│   ├── run_match.py                      # Table 2, Tables S6-S7
│   ├── run_teardown.py                   # Table 3
│   ├── run_coverage.py                   # Figure 4, Tables S10-S11
│   ├── run_tape.py                       # SS and Fluorescence: Fig. 3, Tables S8-S9
│   └── run_contact.py                    # Contact Prediction: Figure 3, Tables S8-S9
├── reproduce/                            # checking a rerun against the paper
│   ├── expected_parity.json              # published PARITY/MAJORITY cells, mean and sd over seeds
│   ├── expected_match.json               # published MATCH cells, mean and sd over seeds
│   ├── make_expected.py                  # derives the two files above from the runs
│   ├── verify.py                         # compares a results file with the published cells
│   ├── make_tables.py                    # rebuilds every supplementary table (S4-S11)
│   └── tape_data.py                      # TAPE and contact loader used by make_tables.py
├── results/
│   └── verification/                     # reruns used to verify this package
│       ├── parity_d64_depth1.json        # PARITY/MAJORITY k=3-5, d=64, depth 1, 3 seeds
│       └── match3_N6_d16_d32.json        # MATCH3, N=6, d=16 and 32, 3 seeds
└── tests/                                # 56 tests, about 30 s on a laptop
    ├── conftest.py
    ├── test_mechanisms.py                # every arm builds; published parameter counts
    ├── test_parity.py                    # offsets, labels, boundary masking, chance level
    ├── test_match.py                     # labels vs brute force; published moduli
    ├── test_linear_value.py              # ablation module equals HOMA when nothing is ablated
    ├── test_contact.py                   # contact parameter counts, symmetry, precision
    ├── test_tape_compat.py               # tokenizer matches TAPE
    └── test_exact_reproduction.py        # retrains two published cells and checks them
```

Runners write their outputs to `results/`. Only `results/verification/` is kept in
the repository.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # torch, numpy, scipy, lmdb, matplotlib, pytest
pytest                           # 56 tests
```

The synthetic experiments need only `torch` and `numpy` (`pip install -e .`).
The protein experiments also need `scipy` and `lmdb` (`pip install -e ".[tape]"`).

`tape_proteins` is **not** needed. It no longer imports under NumPy 2, so the two
pieces of it this code used, the IUPAC tokenizer and the LMDB reader, are
reproduced in `homa/data/tape_compat.py` with identical behaviour.

Tested with Python 3.12 under torch 2.7.1 and torch 2.14 (NumPy 2.2 and 2.5).

## What produces what

| Paper artifact | Command | Cost |
|---|---|---|
| **Table 1**: PARITY-k by order, depth and width; Table S4; Figure 2 | `python experiments/run_parity.py` | 720 runs; ~30 s each at depth 1 on a laptop GPU, ×depth beyond |
| PARITY-3 across widths (Table S5) | `python experiments/run_parity.py --preset capacity` | 60 runs |
| Pairwise-2D at 120 epochs (no longer in the supplement) | `python experiments/run_parity.py --preset long` | 9 runs |
| **Table 2**: MATCH3 by length and width; Tables S6–S7 | `python experiments/run_match.py` | 144 runs; ~30 s each on a T4, ~5 min on a laptop GPU |
| MATCH2 control (Table S6) | `python experiments/run_match.py --orders 2` | |
| **Table 3**: component teardown | `python experiments/run_teardown.py` | 96 runs |
| **Figure 4**: window coverage; Tables S10–S11 | `python experiments/run_coverage.py` | 225 runs, ~10 s each on CPU |
| **Figure 3**: Secondary Structure and Fluorescence; Tables S8–S9 | `python experiments/run_tape.py --data-root DIR` | GPU; hours per width |
| **Figure 3**: Contact Prediction; Tables S8–S9 | `python experiments/run_contact.py --data DIR` | ~21 A100-hours for the full grid |

Every runner:

* writes to `results/<name>.json` after **each** run, and on restart skips
  what it has already done, so an interrupted session loses at most one run;
* refuses to append to a results file written under a different protocol;
* takes `--quick` for a short end-to-end check (a small budget; the numbers are
  not the published ones), and `--tables-only` to print the tables from an
  existing file;
* takes `--seeds`, `--device`, and axis flags (`--widths`, `--orders`, …) to run
  part of a grid.

Result keys match the original result files. A published file can be passed as
`--out` with `--tables-only` to print the published tables, or used as a
starting point that a rerun extends.

## Checking a rerun against the paper

```bash
python experiments/run_parity.py --orders 3,4,5 --depths 1 --widths 64
python reproduce/verify.py --parity results/parity.json

python experiments/run_match.py --lengths 6 --widths 16,32
python reproduce/verify.py --match results/match.json
```

### Rebuilding every supplementary table

`reproduce/make_tables.py` is the script that generated the supplement's table
bodies (Tables S4–S11), with its inputs made configurable. By default it reads
the files the runners write under `results/`:

```bash
python reproduce/make_tables.py > tables.tex                  # from your reruns
python reproduce/make_tables.py --published DIR > tables.tex  # from the original files
```

Run on the original result files, it reproduces all 10 supplementary table
blocks byte for byte. The memory and throughput columns (S9) are only
meaningful for runs on CUDA, since peak memory is recorded only there.

### Checking against reference cells

`reproduce/expected_parity.json` and `reproduce/expected_match.json` hold the
published cells (mean and sample standard deviation over three seeds). They
were computed from the original run records by `reproduce/make_expected.py`,
not typed in.

`verify.py` compares seed means with a tolerance (default 0.02). It sorts each
disagreement into one of two kinds:

* **drift**: the cell moved but stayed on the same side of the line the paper
  draws. A cell at chance is still at chance, and a solved cell is still solved.
* **conflict**: the cell crossed that line. This would contradict a claim, and it
  is the only kind of disagreement counted as a failure.

### How close a rerun gets

Seeding makes a run repeatable on one machine with one torch version. It does
not make runs identical across machines. On these tasks a model often sits
right at the edge of solving the problem, so small numerical differences can
grow:

* **CPU, torch 2.7.1**: the Table 3 and Table S10 cells tested reproduce the
  published records exactly, to every printed digit.
* **CPU, torch 2.14**: the same cells agree to within 4 of 8,000 test labels.
* **Apple MPS**: Table 1 cells agree to within the tolerance on 9 of 12
  cells. The other 3 drift. At PARITY-5 depth 1, Blockwise-3D and HOMA-add
  land at 0.886 and 0.882 against a published 0.938 and 0.948. They are still
  well above chance and below HOMA, which is 1.000 in both.

Compare seed means, not single runs.

## Protocol

These are the settings that produced the published numbers. The runners
hard-code them as `PUBLISHED_CFG` or `PUBLISHED`.

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

* **No FFN in the synthetic models.** A feed-forward sublayer can compute parity
  by itself and would hide what the attention contributes.
* **Residual connections.** Table 1 builds models with residual connections at
  every depth, including depth 1. Table 3 (parity half) uses a single attention
  layer with no residual path. Their one-layer cells are therefore different
  models.
* **Pairwise-2D in the synthetic tables** is run as `blockwise2d` with one block
  spanning the whole sequence, which is the same operator.
* **MATCH modulus.** `M` is pinned per (order, N) in `PUBLISHED_M`, and the
  runner checks at startup that the calibration still reproduces it. The
  embedding is a frozen Fourier basis with a learned projection on top.
* **Fluorescence length is 237**, the longest GFP sequence. The regression head
  reads `max_len × d_model` features, so this setting changes the model, not
  just the padding. The published parameter counts match 237 at every width and
  arm, and none match 512.

## Data

* **Secondary Structure, Fluorescence**: TAPE LMDBs, in one folder per task
  under `--data-root` (or `$TAPE_DATA_DIR`). File names may be
  `<task>_<split>.lmdb` or `<split>.lmdb`.
* **Contact prediction**: the TAPE ProteinNet LMDBs. Point `--data` (or
  `$PROTEINNET_DIR`) at the folder containing `proteinnet_train.lmdb`. The
  loader checks that coordinates are in Ångströms before thresholding at 8 Å.
  An 8.0 threshold on picometre coordinates would fail silently, not loudly.

The TAPE download mirror currently returns HTTP 403, so the data has to come
from an existing copy.

## Licence

Apache 2.0. See `LICENSE`.
