from __future__ import annotations

import numpy as np

from utils.data import (
    DIAGNOSIS_DEATH_PROXY,
    PROCEDURE_DEATH_PROXY,
    assemble_dataset,
    filter_icd10_and_category,
    load_admissions,
    load_code_descriptions,
    load_diagnoses,
    load_procedures,
    split_indices_4way,
)


def test_load_tables_have_expected_columns(synthetic_data_dir):
    admissions = load_admissions(synthetic_data_dir)
    diagnoses = load_diagnoses(synthetic_data_dir)
    procedures = load_procedures(synthetic_data_dir)

    assert set(admissions.columns) == {"hadm_id", "hospital_expire_flag"}
    assert set(diagnoses.columns) == {"hadm_id", "icd_code", "icd_version"}
    assert set(procedures.columns) == {"hadm_id", "icd_code", "icd_version"}


def test_filter_icd10_and_category_format(synthetic_diagnoses):
    out = filter_icd10_and_category(synthetic_diagnoses, prefix="D")
    assert out["category"].str.startswith("D_").all()
    assert (out["category"].str.len() == 5).all()


def test_split_indices_4way_ratios():
    y = np.array([0] * 800 + [1] * 200)
    splits = split_indices_4way(y, seed=1)
    assert len(splits["train"]) == 600
    assert len(splits["val"]) == 100
    assert len(splits["cal"]) == 100
    assert len(splits["test"]) == 200


def test_assemble_dataset_no_proxy_codes(synthetic_data_dir):
    bundle = assemble_dataset(synthetic_data_dir, min_count=1, seed=2)
    banned = {f"D_{c}" for c in DIAGNOSIS_DEATH_PROXY} | {
        f"P_{c}" for c in PROCEDURE_DEATH_PROXY
    }
    assert all(feature not in banned for feature in bundle.feature_names)


def test_assemble_dataset_binary_matrix(synthetic_data_dir):
    bundle = assemble_dataset(synthetic_data_dir, min_count=1, seed=3)
    values = np.unique(bundle.X.data)
    assert set(values.tolist()).issubset({1.0})
    assert bundle.X.shape[0] == bundle.y.shape[0]


def test_load_code_descriptions_prefixes(synthetic_data_dir):
    desc = load_code_descriptions(synthetic_data_dir)
    assert desc["category"].str.startswith(("D_", "P_")).all()
    assert desc["long_title"].notna().all()
