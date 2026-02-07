# MIMIC-IV v3.1 quick schema notes

Data root used in this project:

- `/home/rl/mimic-iv-3.1/mimic-iv-3.1/hosp/`

## Core files for Assignment 1

- `admissions.csv.gz`
  - required columns: `hadm_id`, `hospital_expire_flag`
- `diagnoses_icd.csv.gz`
  - required columns: `hadm_id`, `icd_code`, `icd_version`
- `procedures_icd.csv.gz`
  - required columns: `hadm_id`, `icd_code`, `icd_version`

## Description lookup files

- `d_icd_diagnoses.csv.gz`
  - required columns: `icd_code`, `icd_version`, `long_title`
- `d_icd_procedures.csv.gz`
  - required columns: `icd_code`, `icd_version`, `long_title`

## Assignment-specific transformations

1. Keep only ICD-10 rows (`icd_version == 10`).
2. Build category codes using first 3 characters of `icd_code`.
3. Prefix categories to avoid collisions:
   - diagnosis -> `D_XXX`
   - procedure -> `P_XXX`
4. Drop low-frequency categories (`<10` admissions).
5. Drop death-proxy category lists from assignment spec.

## Split design used in this repo

- Train: 60%
- Validation: 10%
- Calibration: 10% (temperature scaling)
- Test: 20%

All splits are stratified by `hospital_expire_flag`.
