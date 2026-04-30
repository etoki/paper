# MHLW Labor Force Survey 2022 — Data Acquisition Guide

This file documents how to acquire the MHLW Labor Force Survey 2022 marginal
counts that unlock Stage 1 (population aggregation) of the harassment
microsimulation pipeline. It applies to v2.0 of the pre-registration
(OSF DOI [10.17605/OSF.IO/3Y54U](https://osf.io/3y54u)) and is gated on
manual data acquisition per the m8 limitation note.

## What Stage 1 needs

Stage 1 reweights cluster × gender cells using the **gender marginal**
of the working-age population in Japan. The 7-cluster proportions remain
fixed at IEEE-published values (M3-fixed; cluster membership is not in
the MHLW dataset).

Per `code/utils_io.load_mhlw_weights`, the expected CSV schema is:

| column      | type   | required | notes                                          |
|-------------|--------|----------|------------------------------------------------|
| age_group   | str    | yes      | e.g. `15-19`, `20-24`, ..., `65+`              |
| gender      | int/str| yes      | 0/1, F/M, 女/男, female/male — auto-canonicalized |
| count       | int    | yes      | persons (unit consistent across rows)          |
| employment  | str    | optional | `regular`/`non-regular`/...                    |

## Acquisition steps

1. Visit e-Stat: <https://www.e-stat.go.jp/dbview?sid=0003410173>
   (「労働力調査 基本集計 2022年平均」 — Labor Force Survey Basic
   Tabulation 2022 Annual Average).
2. Download the age × gender × employment status crosstab (XLSX or CSV).
3. Reshape to long form matching the schema above. The pipeline accepts
   either Japanese (女/男) or Western (F/M, female/male) gender labels.
4. Save to: `simulation/data/mhlw_labor_force_2022.csv`.

## Verification

Once saved, Stage 1 picks up the file automatically:

```bash
make stage1
# or, with explicit override:
python -m code.stage1_population_aggregation \
    --mhlw-data simulation/data/mhlw_labor_force_2022.csv
```

Without the file, Stage 1 falls back to placeholder gender proportions
[0.5, 0.5] with an explicit warning. The output HDF5 records the
provenance in metadata (`weight_construction` attribute).

## Pre-registration gating

Per v2.0 master Section 5.3 + Methods Clarifications Log Section 5.1
(m8), the cluster proportion source remains the IEEE-published 7-cluster
clustering of N=13,668 (M3-fixed parameters). Only the gender marginal
is updated by MHLW post-stratification in v2.0; full age × gender ×
employment post-stratification is reserved for future work and is not
in scope for the current registered analysis.

The MHLW data acquisition does NOT alter any pre-registered analysis
choices. Acquiring the file simply replaces the placeholder
`[0.5, 0.5]` gender proportions with the actual Japan-2022 working-age
gender marginal.

## File location and version control

The acquired CSV file is small (≪ 1 MB) and is committed to the repository
once acquired. Provenance (download date + e-Stat URL + sheet hash) should
be recorded in this README upon commit so the source is traceable from
GitHub history alone.
