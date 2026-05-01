# HEXACO Harassment Microsimulation — Makefile
#
# Operates under v2.0 master (DOI 10.17605/OSF.IO/3Y54U) + Methods
# Clarifications Log v1.0 (locked 2026-04-30).
#
# Random seed 20260429 is hard-coded across all stages (see
# code/utils_io.py).

PYTHON       := uv run python
PYTEST       := uv run pytest
RUFF         := uv run ruff
SEED         := 20260429

# Output paths
OUT_TABLES   := output/tables
OUT_FIGURES  := output/figures
OUT_SUPP     := output/supplementary

# Source data (read-only)
RAW_DATA     := ../harassment/raw.csv
CENTROIDS    := ../clustering/csv/clstr_kmeans_7c.csv

# Stage outputs (sentinels)
S0_TYPE      := $(OUT_SUPP)/stage0_type_assignment.h5
S0_CELL      := $(OUT_SUPP)/stage0_cell_propensity.h5
S0_EB        := $(OUT_SUPP)/stage0_eb_shrinkage.h5
S1_AGG       := $(OUT_SUPP)/stage1_population_aggregation.h5
S2_VAL       := $(OUT_SUPP)/stage2_validation.h5
S3_SENS      := $(OUT_SUPP)/stage3_sensitivity.h5
S4_BASE      := $(OUT_SUPP)/stage4_baselines.h5
S5_CMV       := $(OUT_SUPP)/stage5_cmv_diagnostic.h5
S6_TRIAL     := $(OUT_SUPP)/stage6_target_trial.h5
S7_COUNTER   := $(OUT_SUPP)/stage7_counterfactual.h5
S8_TRANS     := $(OUT_SUPP)/stage8_transportability.h5
REPORT       := $(OUT_TABLES)/h1_classification.txt

.PHONY: all reproduce verify test lint clean help \
        stage0 stage1 stage2 stage3 stage4 stage5 stage6 stage7 stage8 \
        report

# Default: full reproduction
all: reproduce

# ====================================================================
# Headline target: full pipeline reproduction in ~30 minutes
# ====================================================================

reproduce: $(REPORT)
	@echo ""
	@echo "============================================================"
	@echo "  HEXACO Harassment Microsim reproduction COMPLETE"
	@echo "============================================================"
	@echo "  v2.0 OSF DOI: 10.17605/OSF.IO/3Y54U"
	@echo "  Methods Clarifications Log: v1.0 (locked 2026-04-30)"
	@echo "  Random seed: $(SEED)"
	@echo ""
	@echo "  Headline H1 tier classification:"
	@cat $(REPORT)
	@echo ""
	@echo "  Tables:        $(OUT_TABLES)/"
	@echo "  Figures:       $(OUT_FIGURES)/"
	@echo "  Supplementary: $(OUT_SUPP)/"
	@echo ""

# ====================================================================
# Verification: hash-compare against published reference outputs
# ====================================================================

verify:
	$(PYTHON) code/verify_reproduction.py \
		--output-dir output \
		--reference-hashes output/reference_hashes.json \
		--strict

# ====================================================================
# Per-stage targets (with dependency tracking)
# ====================================================================

stage0: $(S0_TYPE) $(S0_CELL) $(S0_EB)

$(S0_TYPE): code/stage0_type_assignment.py code/utils_io.py $(RAW_DATA) $(CENTROIDS)
	@echo "[Stage 0] Type assignment (hard NN + soft NN sensitivity per M2)"
	$(PYTHON) -m code.stage0_type_assignment --seed $(SEED) --output $@

$(S0_CELL): code/stage0_cell_propensity.py code/utils_bootstrap.py $(S0_TYPE)
	@echo "[Stage 0] 14-cell binary propensity + B=2,000 bootstrap BCa CI (with M4 fallback)"
	$(PYTHON) -m code.stage0_cell_propensity --seed $(SEED) --output $@

$(S0_EB): code/stage0_eb_shrinkage.py code/utils_diagnostics.py $(S0_CELL)
	@echo "[Stage 0] 28-cell EB shrinkage (M1: per-iter (alpha,beta) re-est; m1-m2: MoM rules)"
	$(PYTHON) -m code.stage0_eb_shrinkage --seed $(SEED) --output $@

stage1: $(S1_AGG)

$(S1_AGG): code/stage1_population_aggregation.py $(S0_CELL)
	@echo "[Stage 1] Population aggregation via MHLW Labor Force weights"
	$(PYTHON) -m code.stage1_population_aggregation --seed $(SEED) --output $@

stage2: $(S2_VAL)

$(S2_VAL): code/stage2_validation.py $(S1_AGG)
	@echo "[Stage 2] Validation triangulation + 4-tier classification (m3: B=10,000)"
	$(PYTHON) -m code.stage2_validation --seed $(SEED) --output $@

stage3: $(S3_SENS)

$(S3_SENS): code/stage3_sensitivity.py $(S2_VAL)
	@echo "[Stage 3] Sensitivity sweeps (V/f1/f2/EB scale/threshold/K/role)"
	$(PYTHON) -m code.stage3_sensitivity --seed $(SEED) --output $@

stage4: $(S4_BASE)

$(S4_BASE): code/stage4_baselines.py $(S0_TYPE) $(S0_CELL) $(S1_AGG)
	@echo "[Stage 4] B0-B4 baseline hierarchy + Bonferroni-Holm + Page's L (n4)"
	$(PYTHON) -m code.stage4_baselines --seed $(SEED) --output $@

stage5: $(S5_CMV)

$(S5_CMV): code/stage5_cmv_diagnostic.py
	@echo "[Stage 5] CMV diagnostic (Harman + marker variable)"
	$(PYTHON) -m code.stage5_cmv_diagnostic --seed $(SEED) --output $@

stage6: $(S6_TRIAL)

$(S6_TRIAL): code/stage6_target_trial.py
	@echo "[Stage 6] Target trial emulation specification report"
	$(PYTHON) -m code.stage6_target_trial --seed $(SEED) --output $@

stage7: $(S7_COUNTER)

$(S7_COUNTER): code/stage7_counterfactual.py $(S2_VAL) $(S6_TRIAL)
	@echo "[Stage 7] Counterfactual A/B/C + positivity (m5) + H7 IUT (m7)"
	$(PYTHON) -m code.stage7_counterfactual --seed $(SEED) --output $@

stage8: $(S8_TRANS)

$(S8_TRANS): code/stage8_transportability.py $(S7_COUNTER)
	@echo "[Stage 8] Transportability factor sweep"
	$(PYTHON) -m code.stage8_transportability --seed $(SEED) --output $@

# ====================================================================
# Reporting: aggregate all stage outputs into final tables/figures
# ====================================================================

report: $(REPORT) figures

$(REPORT): code/report.py $(S2_VAL) $(S3_SENS) $(S4_BASE) $(S5_CMV) $(S7_COUNTER) $(S8_TRANS)
	@echo "[Report] Aggregating outputs into tables/figures/supplementary"
	$(PYTHON) -m code.report --output-tier $@

figures: $(S0_CELL) $(S1_AGG) $(S3_SENS) $(S7_COUNTER) $(S8_TRANS)
	@echo "[Figures] Generating Figures 1-6 in PNG/PDF/SVG formats"
	$(PYTHON) -m code.figures \
		--cell-propensity $(S0_CELL) \
		--aggregation $(S1_AGG) \
		--sensitivity $(S3_SENS) \
		--counterfactual $(S7_COUNTER) \
		--transportability $(S8_TRANS) \
		--output-dir $(OUT_FIGURES)

# ====================================================================
# Development targets
# ====================================================================

test:
	$(PYTEST) tests/ -v --cov=code --cov-report=term-missing

lint:
	$(RUFF) check code/ tests/
	$(RUFF) format --check code/ tests/

format:
	$(RUFF) format code/ tests/

# ====================================================================
# Cleanup
# ====================================================================

clean:
	rm -rf $(OUT_SUPP)/*.h5 $(OUT_TABLES)/* $(OUT_FIGURES)/*
	rm -f $(REPORT)

clean-all: clean
	rm -rf .pytest_cache .ruff_cache __pycache__ code/__pycache__ tests/__pycache__

# ====================================================================
# Help
# ====================================================================

help:
	@echo "HEXACO Harassment Microsim — Makefile targets"
	@echo ""
	@echo "  make reproduce      Run full Stage 0 -> 8 pipeline + reporting (~30 min)"
	@echo "  make verify         Hash-compare local outputs vs reference_hashes.json"
	@echo "  make stage0..8      Run a specific stage"
	@echo "  make report         Generate final tables / figures from existing stage outputs"
	@echo "  make test           Run unit tests with coverage"
	@echo "  make lint           Run ruff linter"
	@echo "  make format         Auto-format with ruff"
	@echo "  make clean          Remove generated outputs"
	@echo "  make clean-all      Remove outputs + caches"
	@echo ""
	@echo "  Pre-registration: v2.0 OSF DOI 10.17605/OSF.IO/3Y54U"
	@echo "  Methods Clarifications Log: v1.0 (locked 2026-04-30)"
	@echo "  Random seed: $(SEED)"
