from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


DIAGNOSIS_DEATH_PROXY = {
    "I46",
    "I60",
    "I61",
    "I62",
    "R09",
    "R57",
    "R96",
    "R97",
    "R98",
    "R99",
    "Z66",
    "Z51",
    "Z71",
    "G93",
    "J80",
    "J96",
}

PROCEDURE_DEATH_PROXY = {"5A1", "0BH", "0W9"}


@dataclass(frozen=True)
class DatasetBundle:
    X: csr_matrix
    y: np.ndarray
    hadm_ids: np.ndarray
    feature_names: list[str]
    splits: dict[str, np.ndarray]


def _read_csv(path: Path, usecols: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip", usecols=usecols)


def load_admissions(data_dir: str | Path) -> pd.DataFrame:
    base = Path(data_dir)
    return _read_csv(base / "admissions.csv.gz", ["hadm_id", "hospital_expire_flag"])


def load_diagnoses(data_dir: str | Path) -> pd.DataFrame:
    base = Path(data_dir)
    cols = ["hadm_id", "icd_code", "icd_version"]
    return _read_csv(base / "diagnoses_icd.csv.gz", cols)


def load_procedures(data_dir: str | Path) -> pd.DataFrame:
    base = Path(data_dir)
    cols = ["hadm_id", "icd_code", "icd_version"]
    return _read_csv(base / "procedures_icd.csv.gz", cols)


def filter_icd10_and_category(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    data = frame.loc[frame["icd_version"] == 10, ["hadm_id", "icd_code"]].copy()
    data["icd_code"] = data["icd_code"].astype(str).str.upper().str.strip()
    data["category"] = prefix + "_" + data["icd_code"].str[:3]
    return data[["hadm_id", "category"]]


def drop_rare_categories(events: pd.DataFrame, min_count: int = 10) -> pd.DataFrame:
    counts = (
        events[["hadm_id", "category"]].drop_duplicates().groupby("category").size()
    )
    keep = counts[counts >= min_count].index
    return events[events["category"].isin(keep)].copy()


def drop_death_proxy(events: pd.DataFrame) -> pd.DataFrame:
    banned = {f"D_{c}" for c in DIAGNOSIS_DEATH_PROXY} | {
        f"P_{c}" for c in PROCEDURE_DEATH_PROXY
    }
    return events[~events["category"].isin(banned)].copy()


def build_feature_matrix(
    events: pd.DataFrame, hadm_ids: np.ndarray
) -> tuple[csr_matrix, list[str]]:
    clean = events[["hadm_id", "category"]].drop_duplicates().copy()
    row_lookup = {int(hadm): idx for idx, hadm in enumerate(hadm_ids)}
    cats = sorted(clean["category"].unique().tolist())
    col_lookup = {cat: idx for idx, cat in enumerate(cats)}

    rows = clean["hadm_id"].map(row_lookup).to_numpy()
    cols = clean["category"].map(col_lookup).to_numpy()
    values = np.ones(rows.shape[0], dtype=np.float32)
    matrix = csr_matrix(
        (values, (rows, cols)), shape=(len(hadm_ids), len(cats)), dtype=np.float32
    )
    matrix.data[:] = 1.0
    return matrix, cats


def split_indices_4way(
    y: np.ndarray,
    *,
    train_size: float = 0.60,
    val_size: float = 0.10,
    cal_size: float = 0.10,
    test_size: float = 0.20,
    seed: int = 826,
) -> dict[str, np.ndarray]:
    total = train_size + val_size + cal_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError("Split sizes must sum to 1.0")

    all_idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        all_idx,
        train_size=train_size,
        stratify=y,
        random_state=seed,
    )

    temp_y = y[temp_idx]
    temp_total = val_size + cal_size + test_size
    val_ratio = val_size / temp_total
    cal_ratio = cal_size / temp_total
    test_ratio = test_size / temp_total

    val_idx, caltest_idx = train_test_split(
        temp_idx,
        train_size=val_ratio,
        stratify=temp_y,
        random_state=seed + 1,
    )
    caltest_y = y[caltest_idx]
    cal_idx, test_idx = train_test_split(
        caltest_idx,
        train_size=cal_ratio / (cal_ratio + test_ratio),
        stratify=caltest_y,
        random_state=seed + 2,
    )

    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "cal": np.sort(cal_idx),
        "test": np.sort(test_idx),
    }


def build_split_masks(
    n_rows: int, splits: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name, indices in splits.items():
        mask = np.zeros(n_rows, dtype=bool)
        mask[indices] = True
        masks[name] = mask
    return masks


def load_code_descriptions(data_dir: str | Path) -> pd.DataFrame:
    base = Path(data_dir)
    d_diag = _read_csv(
        base / "d_icd_diagnoses.csv.gz", ["icd_code", "icd_version", "long_title"]
    )
    d_proc = _read_csv(
        base / "d_icd_procedures.csv.gz", ["icd_code", "icd_version", "long_title"]
    )

    d_diag = d_diag[d_diag["icd_version"] == 10].copy()
    d_diag["category"] = "D_" + d_diag["icd_code"].astype(str).str.upper().str[:3]
    d_proc = d_proc[d_proc["icd_version"] == 10].copy()
    d_proc["category"] = "P_" + d_proc["icd_code"].astype(str).str.upper().str[:3]

    merged = pd.concat(
        [d_diag[["category", "long_title"]], d_proc[["category", "long_title"]]],
        ignore_index=True,
    )
    merged["long_title"] = merged["long_title"].fillna("Unknown")
    merged = merged.groupby("category", as_index=False)["long_title"].first()
    return merged


def assemble_dataset(
    data_dir: str | Path, min_count: int = 10, seed: int = 826
) -> DatasetBundle:
    admissions = load_admissions(data_dir)
    diagnoses = load_diagnoses(data_dir)
    procedures = load_procedures(data_dir)

    diag_events = filter_icd10_and_category(diagnoses, prefix="D")
    proc_events = filter_icd10_and_category(procedures, prefix="P")
    events = pd.concat([diag_events, proc_events], ignore_index=True)
    events = drop_rare_categories(events, min_count=min_count)
    events = drop_death_proxy(events)

    labels = admissions[["hadm_id", "hospital_expire_flag"]].drop_duplicates().copy()
    labels = labels.sort_values("hadm_id").reset_index(drop=True)
    hadm_ids = labels["hadm_id"].to_numpy(dtype=np.int64)
    y = labels["hospital_expire_flag"].to_numpy(dtype=np.float32)

    events = events[events["hadm_id"].isin(hadm_ids)]
    X, feature_names = build_feature_matrix(events, hadm_ids)
    splits = split_indices_4way(y, seed=seed)

    return DatasetBundle(
        X=X, y=y, hadm_ids=hadm_ids, feature_names=feature_names, splits=splits
    )
