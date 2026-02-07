from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(826)


@pytest.fixture
def synthetic_admissions(rng: np.random.Generator) -> pd.DataFrame:
    n = 200
    hadm_ids = np.arange(1, n + 1)
    y = rng.binomial(1, 0.2, size=n)
    return pd.DataFrame({"hadm_id": hadm_ids, "hospital_expire_flag": y})


@pytest.fixture
def synthetic_diagnoses(
    rng: np.random.Generator, synthetic_admissions: pd.DataFrame
) -> pd.DataFrame:
    hadm_ids = synthetic_admissions["hadm_id"].to_numpy()
    n_rows = 800
    codes = np.array(["A100", "B200", "C300", "I469", "J961", "Z660", "D500"])
    versions = rng.choice([9, 10], size=n_rows, p=[0.25, 0.75])
    return pd.DataFrame(
        {
            "hadm_id": rng.choice(hadm_ids, size=n_rows),
            "icd_code": rng.choice(codes, size=n_rows),
            "icd_version": versions,
        }
    )


@pytest.fixture
def synthetic_procedures(
    rng: np.random.Generator, synthetic_admissions: pd.DataFrame
) -> pd.DataFrame:
    hadm_ids = synthetic_admissions["hadm_id"].to_numpy()
    n_rows = 600
    codes = np.array(["0BH7", "5A12", "0W90", "1AA1", "2BB2", "3CC3"])
    versions = rng.choice([9, 10], size=n_rows, p=[0.30, 0.70])
    return pd.DataFrame(
        {
            "hadm_id": rng.choice(hadm_ids, size=n_rows),
            "icd_code": rng.choice(codes, size=n_rows),
            "icd_version": versions,
        }
    )


@pytest.fixture
def synthetic_data_dir(
    tmp_path: Path,
    synthetic_admissions: pd.DataFrame,
    synthetic_diagnoses: pd.DataFrame,
    synthetic_procedures: pd.DataFrame,
) -> Path:
    synthetic_admissions.to_csv(
        tmp_path / "admissions.csv.gz", index=False, compression="gzip"
    )
    synthetic_diagnoses.to_csv(
        tmp_path / "diagnoses_icd.csv.gz", index=False, compression="gzip"
    )
    synthetic_procedures.to_csv(
        tmp_path / "procedures_icd.csv.gz", index=False, compression="gzip"
    )

    d_diag = pd.DataFrame(
        {
            "icd_code": ["A100", "B200", "C300", "I469", "J961", "Z660", "D500"],
            "icd_version": [10, 10, 10, 10, 10, 10, 10],
            "long_title": [
                "Code A",
                "Code B",
                "Code C",
                "Cardiac arrest",
                "Respiratory failure",
                "Do not resuscitate",
                "Code D",
            ],
        }
    )
    d_proc = pd.DataFrame(
        {
            "icd_code": ["0BH7", "5A12", "0W90", "1AA1", "2BB2", "3CC3"],
            "icd_version": [10, 10, 10, 10, 10, 10],
            "long_title": [
                "Intubation",
                "Ventilation",
                "Drain",
                "Proc1",
                "Proc2",
                "Proc3",
            ],
        }
    )
    d_diag.to_csv(tmp_path / "d_icd_diagnoses.csv.gz", index=False, compression="gzip")
    d_proc.to_csv(tmp_path / "d_icd_procedures.csv.gz", index=False, compression="gzip")
    return tmp_path
