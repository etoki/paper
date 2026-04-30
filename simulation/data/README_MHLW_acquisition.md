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

## Acquired (2026-04-30): mhlw_labor_force_2022.csv

**Source PDF**: `労働力調査（基本集計）2022年（令和４年）平均結果の要約.pdf`
- Publisher: 総務省統計局 (Ministry of Internal Affairs and Communications, Statistics Bureau)
- Publication date: 令和5年1月31日 (2023-01-31)
- e-Stat metadata: `労働力調査（基本集計）2022年（令和４年）平均結果の要約、概要、統計表等`
- Page extracted: p.4 表3「年齢階級別就業者数の推移」(Table 3: Annual employed persons by age group)

**Population chosen**: 就業者 (employed persons), 2022 annual average

| 区分 | 男女計 | 男 | 女 |
|------|-------:|----:|----:|
| 総数（万人） | 6,723 | 3,699 | 3,024 |
| 15-64歳    | 5,810 | 3,161 | 2,649 |
| 65歳以上    |   912 |   538 |   375 |

**Resulting marginals**:
- F (女) = 3,024 / 6,723 = **0.4498** (44.98%)
- M (男) = 3,699 / 6,723 = **0.5502** (55.02%)

**Why 就業者 (vs 役員除く雇用者)?**
The MHLW power harassment survey denominator is 「労働者」 (workers in the
broader sense, including self-employed). 就業者 is the closest demographic
match. For sensitivity, 役員除く雇用者 (employees excluding officers, total
5,699万人, F=2,682/47.06%, M=3,017/52.94%) is recorded as an alternative
population definition in the source PDF page 9 表7.

**Long-form CSV schema delivered**:
```csv
age_group,gender,count,employment
15-64,女,2649,employed
65+,女,375,employed
15-64,男,3161,employed
65+,男,538,employed
```

The loader (`utils_io.load_mhlw_weights`) sums across `age_group` and
`employment` to derive the gender marginal used by Stage 1. Age columns
are preserved in the long-form table for future age-stratified
post-stratification (out of scope for v2.0; reserved for Phase 2 spin-off).

**Reproducibility hash**:
```
SHA256 of source PDF: (recorded at git add time via 'git hash-object')
File path: simulation/data/労働力調査（基本集計）2022年（令和４年）平均結果の要約.pdf
```
