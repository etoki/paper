# Simulation projects in `paper/simulation/`

This directory hosts two **independent simulation projects** that share infrastructure:

1. **HEXACO Harassment Microsimulation** (this README's primary subject) — sole-authored secondary analysis + microsimulation + target trial emulation. Pre-registered on OSF with DOI [10.17605/OSF.IO/3Y54U](https://osf.io/3y54u).
2. **Big Five Generative Agent Simulation** (separate project; see [`HANDOFF.md`](HANDOFF.md)) — sole-authored generative-agent simulation of Japanese university entrance outcomes. Independent codebase under `agent/`, `data/`, with its own documentation in `HANDOFF.md` and `deep_reading_notes.md`.

The structure under `simulation/` allocates new files (under Section 8.2 of the Harassment Microsim preregistration) without disturbing the Big Five project's existing files.

---

## HEXACO 7-Typology Workplace Harassment Microsimulation

### Pre-registration

- **Master document** (current primary, v2.0): [OSF DOI 10.17605/OSF.IO/3Y54U](https://osf.io/3y54u) — locally at `docs/pre_registration/D12_pre_registration_OSF.md` (Japanese) and `.en.md` (English).
- **Historical** (superseded, v1.1): [OSF DOI 10.17605/OSF.IO/45QP9](https://osf.io/45qp9). Public diff at `docs/pre_registration/D12_pre_registration_v1.1_to_v2.0_diff.pdf`.
- **Methods Clarifications Log** (Section 6.5 Level 1 deviation, locked 2026-04-30): `docs/pre_registration/D12_v2.0_methods_clarifications.md`.
- **OSF associated project**: [https://osf.io/3hxz6](https://osf.io/3hxz6)

The combined specification is **v2.0 master + Methods Clarifications Log**; both are locked and Stage 0 code execution proceeds under their joint authority.

### Repository structure (per v2.0 master Section 8.2)

```
simulation/
├── docs/
│   ├── pre_registration/                          # all locked preregistration documents
│   ├── notes/                                     # internal research plan v6/v7
│   ├── literature_audit/                          # 60+ paper deep reading
│   └── power_analysis/                            # D13 power analysis
├── code/
│   ├── stage0_type_assignment.py                  # 7-cluster nearest-neighbor classification (M2: hard / soft modes)
│   ├── stage0_cell_propensity.py                  # 14-cell binary outcome propensity + bootstrap CI
│   ├── stage0_eb_shrinkage.py                     # 28-cell EB shrinkage (M1: per-iter (alpha,beta) re-est; m1-m2: MoM rules)
│   ├── stage1_population_aggregation.py           # MHLW Labor Force-weighted national prevalence
│   ├── stage2_validation.py                       # MAPE vs MHLW H28/R2/R5 + 4-tier classification (m3: B=10000)
│   ├── stage3_sensitivity.py                      # V/f1/f2/EB scale/threshold/K/role sweeps
│   ├── stage4_baselines.py                        # B0-B4 hierarchy + Page's L (n4)
│   ├── stage5_cmv_diagnostic.py                   # Harman + marker variable
│   ├── stage6_target_trial.py                     # PICO + identifying assumptions
│   ├── stage7_counterfactual.py                   # ΔP_x with do-operator + positivity (m5) + H7 IUT (m7)
│   ├── stage8_transportability.py                 # cultural-attenuation factor sweep
│   ├── utils_bootstrap.py                         # M4: 4-step CI priority chain (CP -> BCa -> BC -> percentile); m4: jackknife
│   ├── utils_diagnostics.py                       # MoM rejection rules; positivity checks; sanity reports
│   └── utils_io.py                                # data loading / output orchestration
├── tests/
│   └── test_*.py                                  # one test file per stage + utils
├── output/
│   ├── tables/                                    # main + sensitivity tables
│   ├── figures/                                   # calibration plots, distributions
│   └── supplementary/                             # per-cell CI-method log, MoM diagnostics, etc.
├── agent/                                         # ★ Big Five project (separate, do not modify from harassment microsim)
├── data/                                          # ★ Big Five project data (separate)
├── prior_research/                                # PDFs and text extractions (shared)
├── Dockerfile                                     # Python 3.11-slim + dependencies
├── pyproject.toml                                 # uv / pip-compatible
├── uv.lock                                        # generated on first `uv lock`
├── Makefile                                       # `make reproduce` 30-minute target
└── README.md                                      # this file
```

### How to reproduce in 30 minutes

> **Reproducibility commitment** (per v2.0 Section 8 / D-NEW9): all results in the eventual paper can be regenerated end-to-end in 30 minutes on commodity hardware, using only the inputs in `harassment/raw.csv`, `clustering/csv/clstr_kmeans_7c.csv`, and the public MHLW / Pasona PDFs in `simulation/prior_research/_text/`.

#### Prerequisites

- **OS**: Ubuntu 22.04+ / macOS 13+ / WSL2 on Windows 10+
- **Hardware**: 4+ CPU cores, 8+ GB RAM. No GPU required.
- **Software**: one of:
  - Python 3.11+ with [`uv`](https://docs.astral.sh/uv/) (recommended, fast)
  - Or Docker 24+ (use the `Dockerfile`)

#### Quick reproduction (using `uv`)

```bash
# from repo root
cd simulation/

# install dependencies
uv sync

# run the full pipeline (Stage 0 -> Stage 8 + reports)
make reproduce
```

Expected runtime: 25-30 minutes (single core; multi-core scales linearly for the bootstrap phase).

Outputs land in `output/tables/`, `output/figures/`, `output/supplementary/`. The headline H1 tier classification (Strict / Standard SUCCESS / PARTIAL / FAILURE) is printed to stdout and logged to `output/supplementary/h1_classification.txt`.

#### Quick reproduction (using Docker)

```bash
cd simulation/
docker build -t hexaco-harassment-microsim .
docker run --rm -v "$(pwd)/output:/work/output" hexaco-harassment-microsim make reproduce
```

#### What the pipeline does

1. **Stage 0** (~3 min): assigns N=354 to 7 clusters (hard NN per primary; soft NN τ ∈ {0.5, 1.0, 2.0} × median NN per M2 sensitivity); computes 14-cell binary harassment propensities with B=2,000 bootstrap BCa CIs; runs 28-cell EB shrinkage with per-iteration (α̂, β̂) re-estimation per M1.
2. **Stage 1** (~2 min): scales cell propensities to ~68M Japanese workforce via MHLW Labor Force Survey weights.
3. **Stage 2** (~5 min): computes headline national MAPE_FY2016 with B=10,000 (per m3); FY2020 / FY2023 with B=10,000 each; assigns 4-tier classification.
4. **Stage 3** (~8 min): sweeps V, f1, f2, EB scale, binarization threshold, cluster K, role-estimation models.
5. **Stage 4** (~3 min): B0–B4 baseline hierarchy + Page's L auxiliary (n4).
6. **Stage 5** (~1 min): CMV diagnostic on N=13,668 personality data.
7. **Stage 6** (~1 min): target trial emulation specification report.
8. **Stage 7** (~3 min): counterfactual A/B/C ΔP estimates with positivity diagnostic ρ_{c,x} (m5) and H7 IUT (m7).
9. **Stage 8** (~1 min): transportability factor sweep.
10. **Reporting** (~2 min): tier classifications, CI tables, calibration plots, deviation log entries.

#### Random seed

All stochastic operations use `default_rng(seed=20260429)` (per v2.0 Section 2.4). The seed is hard-coded and is NOT a configurable parameter; reproducibility is contingent on this seed.

#### Verifying reproduction

After `make reproduce` completes, run `make verify` to compare your local outputs against the published reference outputs (recorded as SHA256 hashes in `output/reference_hashes.json`). If hashes match, your reproduction is byte-identical to the lock-time results.

### Working specification

The Stage 0–8 implementation follows:

- **v2.0 master document**: structural specification (which stages, which hypotheses, what thresholds)
- **Methods Clarifications Log v1.0**: refinements per anonymous methodologist review (M1–M4 Major + m1–m7 Minor + n1–n5 Nitpicks)

When the two documents disagree (none expected; verified during clarifications log lock), v2.0 master takes precedence at the structural level and the clarifications log refines the implementation level.

### License

- **Code**: MIT (see `code/LICENSE` if present, otherwise inherits from repo root `LICENSE`).
- **Pre-registration documents**: CC-BY 4.0 (per OSF default).
- **Aggregated data outputs**: CC-BY 4.0 in `output/`.
- **Cell-level raw data**: restricted access per v2.0 Section 9.5 (re-identification risk for cells with N < 10).

### Citation

If you use this work, please cite the SocArXiv preprint and the v2.0 OSF registration:

> Tokiwa, E. (2026). *Person-Level versus System-Level Anti-Harassment Interventions: A HEXACO 7-Typology Counterfactual Microsimulation in Japanese Workplaces* [Preprint]. SocArXiv. https://doi.org/10.31235/osf.io/p2d8w_v1

> Tokiwa, E. (2026). HEXACO 7-Typology Workplace Harassment Microsimulation (v2.0). Open Science Framework Pre-Registration. https://doi.org/10.17605/OSF.IO/3Y54U

### Contact

Eisuke Tokiwa, SUNBLAZE Co., Ltd. — eisuke.tokiwa@sunblaze.jp — ORCID 0009-0009-7124-6669

The author commits to maintaining contact and OSF registrations for at least 10 years (per v2.0 Section 9.6 long-term ethical monitoring).

---

## Big Five Generative Agent Simulation (separate project)

See [`HANDOFF.md`](HANDOFF.md) for the Big Five project's documentation, which is independent of the harassment microsim documented above. The Big Five project lives in `agent/` and `data/` and uses its own dependency setup (currently not via `pyproject.toml` at this `simulation/` level — see `HANDOFF.md`).
