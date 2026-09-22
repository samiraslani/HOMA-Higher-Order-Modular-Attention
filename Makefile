# Convenience targets.  Every target is a single documented command; see README.
PY ?= python

.PHONY: install test quick parity parity-capacity parity-long match match2 coverage tape tables

install:
	$(PY) -m pip install -e ".[all]"

test:
	$(PY) -m pytest

# End-to-end pipeline check of every diagnostic runner at a tiny budget (~5 min).
quick:
	$(PY) experiments/run_parity.py   --quick --out results/quick/parity.json
	$(PY) experiments/run_match.py    --quick --out results/quick/match.json
	$(PY) experiments/run_coverage.py --quick --out results/quick/coverage.json

parity:           ; $(PY) experiments/run_parity.py
parity-capacity:  ; $(PY) experiments/run_parity.py --preset capacity
parity-long:      ; $(PY) experiments/run_parity.py --preset long
match:            ; $(PY) experiments/run_match.py
match2:           ; $(PY) experiments/run_match.py --orders 2 --out results/match2.json
coverage:         ; $(PY) experiments/run_coverage.py
tape:             ; $(PY) experiments/run_tape.py --data-root $(TAPE_DATA_DIR)
tables:           ; $(PY) reproduce/make_tables.py > results/tables.tex
