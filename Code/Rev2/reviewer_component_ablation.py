#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reviewer_component_ablation_20train_10test.py

Purpose
-------
Reviewer-focused component/branch evaluation using:
  * DABRA.csv
  * VeReMi.csv
  * 20% train
  * 10% test
  * seeds: 5, 11, 23, 37, 45

Important scientific scope
--------------------------
DABRA and VeReMi are evaluated independently. This script does NOT row-join,
timestamp-align, transfer labels, or fabricate paired DABRA/VeReMi samples.

DABRA:
  target = derived_failure_tau_3
  evaluates driving-risk / decision-validation predictive branches.

VeReMi:
  target = (class != 0)
  evaluates V2X fault/misbehavior prediction.
  The VeReMi predictive model is the experimentally supported SEFN task
  instantiation. Branch-removal variants quantify dependence on position,
  speed, acceleration, and heading.

SCTM:
  SCTM is post-hoc in the manuscript architecture. Removing SCTM cannot
  legitimately change neural AUROC/AUPRC unless SCTM feeds back into the
  predictor. Therefore this script reports predictive metrics separately
  and does NOT invent a "Without_SCTM" predictive AUROC.

Outputs
-------
reviewer_20train_10test/
  all_seed_metrics.csv
  summary_metrics.csv
  best_results.csv
  reviewer_table.csv
  reviewer_table.tex
  split_manifest.csv

Dependencies
------------
pip install numpy pandas scikit-learn scipy
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ================================================================
# USER PATHS
# ================================================================

PROJECT_ROOT = Path(r"E:\other\Causal-DT")
DATASET_DIR = PROJECT_ROOT / "Dataset"
FINAL_RESULTS_DIR = PROJECT_ROOT / "FinalResults_50_10_10"
FINAL_DATA_DIR = FINAL_RESULTS_DIR / "data"
AUDIT_DIR = PROJECT_ROOT / "DatasetAudit"

DABRA_CSV = DATASET_DIR / "DABRA.csv"
VEREMI_CSV = DATASET_DIR / "VeReMi.csv"

# Preferred finalized/audited data artifacts.
DABRA_FINAL_PARQUET = FINAL_DATA_DIR / "dabra_full_with_target.parquet"
DABRA_AUDIT_CSV = AUDIT_DIR / "dabra" / "analysis_sample.csv"
VEREMI_AUDIT_CSV = AUDIT_DIR / "veremi" / "analysis_sample.csv"

OUT = FINAL_RESULTS_DIR / "reviewer_20train_10test"
OUT.mkdir(parents=True, exist_ok=True)

# ================================================================
# PROTOCOL
# ================================================================

SEEDS = [5, 11, 23, 37, 45]
TRAIN_FRAC = 0.20
TEST_FRAC = 0.10
MAX_FSR = 0.05

# VeReMi is ~1.18 GB in the screenshot. Read only needed columns.
VEREMI_FEATURES = {
    "position": ["posx", "posy", "posx_n", "posy_n"],
    "speed": ["spdx", "spdy", "spdx_n", "spdy_n"],
    "acceleration": ["aclx", "acly", "aclx_n", "acly_n"],
    "heading": ["hedx", "hedy", "hedx_n", "hedy_n"],
}

# DABRA feature groups from the manuscript/code specification.
DABRA_FEATURES = {
    "behavior": [
        "latitude", "longitude", "speed", "acceleration", "steering_angle",
        "heading", "trip_duration", "trip_distance", "fuel_consumption",
        "rpm", "brake_usage", "lane_deviation", "route_deviation_score",
        "acceleration_variation", "behavioral_consistency_index",
    ],
    "environment": ["weather_conditions", "road_type", "traffic_condition"],
}

DABRA_TARGET = "derived_failure_tau_3"
VEREMI_CLASS = "class"

# ================================================================
# METRICS
# ================================================================

def ece_score(y, p, bins=15):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)

def adaptive_ece(y, p, bins=15):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    groups = np.array_split(order, bins)
    out = 0.0
    n = len(y)
    for idx in groups:
        if len(idx) == 0:
            continue
        out += len(idx) / n * abs(y[idx].mean() - p[idx].mean())
    return float(out)

def brier_decomposition(y, p, bins=15):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    prevalence = float(y.mean())
    uncertainty = prevalence * (1.0 - prevalence)
    reliability = 0.0
    resolution = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        nk = int(m.sum())
        if nk == 0:
            continue
        wk = nk / len(y)
        pk = float(p[m].mean())
        ok = float(y[m].mean())
        reliability += wk * (pk - ok) ** 2
        resolution += wk * (ok - prevalence) ** 2
    return float(reliability), float(resolution), float(uncertainty)

def threshold_stats(y, p, threshold):
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(p) >= threshold).astype(int)
    pos = y == 1
    neg = y == 0
    fsr = float(((pred == 0) & pos).sum() / max(1, pos.sum()))
    fur = float(((pred == 1) & neg).sum() / max(1, neg.sum()))
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "fsr": fsr,
        "fur": fur,
    }

def choose_threshold(y, p, max_fsr=MAX_FSR):
    # Mirrors the stated manuscript rule:
    # prefer FSR <= .05, then maximize BAcc, then lower FUR.
    candidates = np.unique(
        np.concatenate([
            np.linspace(0.01, 0.99, 199),
            np.quantile(p, np.linspace(0.01, 0.99, 99)),
        ])
    )
    best = None
    for th in candidates:
        s = threshold_stats(y, p, float(th))
        valid = s["fsr"] <= max_fsr
        key = (
            int(valid),
            s["balanced_accuracy"],
            -s["fur"],
            -float(th),
        )
        if best is None or key > best[0]:
            best = (key, float(th))
    return best[1]

def probability_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    rel, res, unc = brier_decomposition(y, p)
    d = {
        "auroc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "auprc": average_precision_score(y, p),
        "prevalence": float(y.mean()),
        "auprc_gain": average_precision_score(y, p) - float(y.mean()),
        "brier": brier_score_loss(y, p),
        "ece": ece_score(y, p),
        "adaptive_ece": adaptive_ece(y, p),
        "nll": log_loss(y, p, labels=[0, 1]),
        "threshold": float(threshold),
        "brier_reliability": rel,
        "brier_resolution": res,
        "brier_uncertainty": unc,
    }
    d.update(threshold_stats(y, p, threshold))
    return d

# ================================================================
# SPLITTING
# ================================================================

def stratified_20_10_indices(y, seed):
    """
    Exact disjoint 20% train and 10% test selection.
    The remaining 70% is not used in this reviewer experiment.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    train, test = [], []

    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_train = int(round(TRAIN_FRAC * len(idx)))
        n_test = int(round(TEST_FRAC * len(idx)))
        train.extend(idx[:n_train])
        test.extend(idx[n_train:n_train + n_test])

    train = np.asarray(train, dtype=int)
    test = np.asarray(test, dtype=int)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test

def split_train_validation(train_idx, y, seed, val_ratio=0.20):
    """
    Reserve 20% of the selected 20%-training subset for threshold selection.
    Therefore no test information is used to choose the threshold.
    """
    rng = np.random.default_rng(seed + 1000)
    y_train = np.asarray(y)[train_idx]
    fit, val = [], []
    for cls in np.unique(y_train):
        local = np.flatnonzero(y_train == cls)
        rng.shuffle(local)
        nv = max(1, int(round(val_ratio * len(local))))
        val.extend(train_idx[local[:nv]])
        fit.extend(train_idx[local[nv:]])
    return np.asarray(fit), np.asarray(val)

# ================================================================
# DATA
# ================================================================

def snake(name):
    """Normalize raw CSV column names exactly enough for the project datasets."""
    import re
    s = str(name).strip()
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    return s.strip("_").lower()


def existing_columns(path):
    return list(pd.read_csv(path, nrows=0).columns)

def load_dabra():
    """
    Load the finalized DABRA artifact when available.

    Priority:
      1. FinalResults_50_10_10/data/dabra_full_with_target.parquet
      2. DatasetAudit/dabra/analysis_sample.csv
      3. Dataset/DABRA.csv

    If the selected source has no stored target, reconstruct the fixed tau=3
    target from the original target-source columns. These source columns are
    never included in the predictive feature groups.
    """
    candidates = [
        ("final parquet", DABRA_FINAL_PARQUET),
        ("audit sample", DABRA_AUDIT_CSV),
        ("raw csv", DABRA_CSV),
    ]

    source_name = None
    source_path = None
    for name, path in candidates:
        if path.exists():
            source_name = name
            source_path = path
            break

    if source_path is None:
        checked = "\n".join(f"  - {p}" for _, p in candidates)
        raise FileNotFoundError(
            "No DABRA input file was found. Checked:\n" + checked
        )

    print(f"[LOAD DABRA] {source_path}")

    if source_path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(source_path)
        except ImportError as exc:
            raise ImportError(
                "Reading the finalized DABRA parquet requires pyarrow. "
                "Install it with: pip install pyarrow"
            ) from exc
    else:
        df = pd.read_csv(source_path, low_memory=False)

    df.columns = [snake(c) for c in df.columns]

    # The finalized runner may store the outcome under either name.
    target_candidates = [
        "target",
        "derived_failure_tau_3",
        "derived_failure",
    ]
    target_col = next(
        (c for c in target_candidates if c in df.columns),
        None,
    )

    if target_col is not None:
        target_numeric = pd.to_numeric(df[target_col], errors="coerce")
        bad = int(target_numeric.isna().sum())
        if bad:
            raise ValueError(
                f"DABRA target '{target_col}' has {bad} non-numeric/missing values."
            )

        unique = set(target_numeric.unique().tolist())
        if not unique.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"DABRA target '{target_col}' is not binary. "
                f"Observed values include: {sorted(unique)[:20]}"
            )

        y = target_numeric.astype(np.int8).to_numpy()
        print(f"[DABRA] stored target: {target_col}")

    else:
        required = [
            "anomalous_event",
            "route_anomaly",
            "geofencing_violation",
            "stop_events",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                "DABRA has no stored target and target reconstruction is "
                f"impossible because these columns are missing: {missing}"
            )

        anomalous = (
            pd.to_numeric(df["anomalous_event"], errors="coerce")
            .fillna(0).astype(int).eq(1)
        )
        route = (
            pd.to_numeric(df["route_anomaly"], errors="coerce")
            .fillna(0).astype(int).eq(1)
        )
        geofence = (
            pd.to_numeric(df["geofencing_violation"], errors="coerce")
            .fillna(0).astype(int).eq(1)
        )
        stops = (
            pd.to_numeric(df["stop_events"], errors="coerce")
            .fillna(0).ge(3)
        )

        y = (
            anomalous | route | geofence | stops
        ).astype(np.int8).to_numpy()

        df["derived_failure_tau_3"] = y
        print("[DABRA] target reconstructed with fixed tau=3")

    n = len(y)
    n_pos = int(y.sum())
    n_neg = int(n - n_pos)

    print(
        f"[DABRA] N={n:,} | positive={n_pos:,} | "
        f"negative={n_neg:,} | prevalence={n_pos / n:.6f}"
    )

    if n == 120000:
        if n_pos != 35180 or n_neg != 84820:
            raise ValueError(
                "DABRA target counts do not match the finalized experiment. "
                f"Observed positive/negative={n_pos}/{n_neg}; "
                "expected 35180/84820. Stop before training."
            )

    return df, y


def load_veremi():
    """
    Load VeReMi independently. The raw CSV is preferred because it preserves
    the original class labels required to construct the V2X fault target.
    """
    if VEREMI_CSV.exists():
        source_path = VEREMI_CSV
    elif VEREMI_AUDIT_CSV.exists():
        source_path = VEREMI_AUDIT_CSV
    else:
        raise FileNotFoundError(
            "No VeReMi input found at:\n"
            f"  - {VEREMI_CSV}\n"
            f"  - {VEREMI_AUDIT_CSV}"
        )

    print(f"[LOAD VEREMI] {source_path}")

    raw_cols = list(pd.read_csv(source_path, nrows=0).columns)
    normalized_to_raw = {snake(c): c for c in raw_cols}

    requested = [VEREMI_CLASS]
    for cols in VEREMI_FEATURES.values():
        requested.extend(cols)
    requested = list(dict.fromkeys(requested))

    missing = [c for c in requested if c not in normalized_to_raw]
    if VEREMI_CLASS in missing:
        raise ValueError(
            "VeReMi class column is missing. Available normalized columns: "
            f"{sorted(normalized_to_raw.keys())}"
        )

    # Feature columns missing from a source are reported explicitly.
    feature_missing = [c for c in missing if c != VEREMI_CLASS]
    if feature_missing:
        print(
            "[VEREMI WARNING] Missing requested feature columns: "
            + ", ".join(feature_missing)
        )

    available = [c for c in requested if c in normalized_to_raw]
    raw_usecols = [normalized_to_raw[c] for c in available]

    df = pd.read_csv(
        source_path,
        usecols=raw_usecols,
        low_memory=False,
    )
    df.columns = [snake(c) for c in df.columns]

    cls = pd.to_numeric(df[VEREMI_CLASS], errors="coerce")
    bad = int(cls.isna().sum())
    if bad:
        raise ValueError(
            f"VeReMi class column contains {bad} non-numeric/missing values."
        )

    cls = cls.astype(int)
    y = (cls != 0).astype(np.int8).to_numpy()

    n = len(y)
    n_pos = int(y.sum())
    n_neg = int(n - n_pos)
    print(
        f"[VEREMI] N={n:,} | attack={n_pos:,} | normal={n_neg:,} | "
        f"prevalence={n_pos / n:.6f}"
    )

    return df, y


# ================================================================
# MODEL
# ================================================================

def build_pipeline(df, features):
    features = [c for c in features if c in df.columns]
    if not features:
        raise ValueError("No requested features exist in dataframe.")

    numeric = [
        c for c in features
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical = [c for c in features if c not in numeric]

    transformers = []

    if numeric:
        num = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", num, numeric))

    if categorical:
        cat = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat, categorical))

    pre = ColumnTransformer(transformers)

    # Fast, reproducible reviewer benchmark.
    clf = LogisticRegression(
        solver="liblinear",
        max_iter=250,
        class_weight=None,
    )

    return Pipeline([
        ("pre", pre),
        ("clf", clf),
    ]), features

def evaluate_variant(dataset, df, y, feature_groups, variant, active_groups, seed):
    all_features = []
    for group in active_groups:
        all_features.extend(feature_groups[group])
    all_features = list(dict.fromkeys(all_features))
    all_features = [c for c in all_features if c in df.columns]

    train_idx, test_idx = stratified_20_10_indices(y, seed)
    fit_idx, val_idx = split_train_validation(train_idx, y, seed)

    pipe, all_features = build_pipeline(df, all_features)

    t0 = time.perf_counter()
    pipe.fit(df.iloc[fit_idx][all_features], y[fit_idx])
    train_seconds = time.perf_counter() - t0

    p_val = pipe.predict_proba(df.iloc[val_idx][all_features])[:, 1]
    threshold = choose_threshold(y[val_idx], p_val)

    t1 = time.perf_counter()
    p_test = pipe.predict_proba(df.iloc[test_idx][all_features])[:, 1]
    inference_seconds = time.perf_counter() - t1
    latency_ms_per_sample = 1000.0 * inference_seconds / max(1, len(test_idx))

    metrics = probability_metrics(y[test_idx], p_test, threshold)
    metrics.update({
        "dataset": dataset,
        "variant": variant,
        "active_groups": "+".join(active_groups),
        "n_features_raw": len(all_features),
        "seed": seed,
        "train_selected_n": len(train_idx),
        "train_fit_n": len(fit_idx),
        "validation_n": len(val_idx),
        "test_n": len(test_idx),
        "training_seconds": train_seconds,
        "latency_ms_per_sample": latency_ms_per_sample,
    })
    return metrics

# ================================================================
# EXPERIMENT DEFINITIONS
# ================================================================

def dabra_variants():
    return {
        "Full_DABRA_CIDT_inputs": ["behavior", "environment"],
        "Without_behavior": ["environment"],
        "Without_environment": ["behavior"],
    }

def veremi_variants():
    # VeReMi is the supported V2X fault/misbehavior / SEFN task.
    return {
        "Full_VeReMi_SEFN_inputs": [
            "position", "speed", "acceleration", "heading"
        ],
        "Without_position": ["speed", "acceleration", "heading"],
        "Without_speed": ["position", "acceleration", "heading"],
        "Without_acceleration": ["position", "speed", "heading"],
        "Without_heading": ["position", "speed", "acceleration"],
    }

# ================================================================
# SUMMARY
# ================================================================

SUMMARY_METRICS = [
    "auroc", "auprc", "auprc_gain", "brier",
    "ece", "adaptive_ece", "nll",
    "accuracy", "balanced_accuracy", "mcc", "f1",
    "fsr", "fur", "threshold",
    "brier_reliability", "brier_resolution", "brier_uncertainty",
    "training_seconds", "latency_ms_per_sample",
]

def t_ci(values):
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(x) < 2:
        return np.nan, np.nan
    mean = x.mean()
    sd = x.std(ddof=1)
    crit = t.ppf(0.975, len(x) - 1)
    margin = crit * sd / math.sqrt(len(x))
    return mean - margin, mean + margin

def summarize(df):
    rows = []
    for (dataset, variant), g in df.groupby(["dataset", "variant"], sort=False):
        row = {
            "dataset": dataset,
            "variant": variant,
            "runs": g["seed"].nunique(),
            "active_groups": g["active_groups"].iloc[0],
            "n_features_raw": g["n_features_raw"].iloc[0],
            "train_selected_n": int(round(g["train_selected_n"].mean())),
            "test_n": int(round(g["test_n"].mean())),
        }
        for m in SUMMARY_METRICS:
            vals = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = vals.mean()
            row[f"{m}_std"] = vals.std(ddof=1)
        for m in ["auroc", "auprc", "brier", "ece", "balanced_accuracy", "mcc", "fsr", "fur"]:
            lo, hi = t_ci(g[m])
            row[f"{m}_ci_low"] = lo
            row[f"{m}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)

def add_deltas(summary):
    out = summary.copy()
    for dataset in out["dataset"].unique():
        part = out["dataset"].eq(dataset)
        if dataset == "DABRA":
            base_name = "Full_DABRA_CIDT_inputs"
        else:
            base_name = "Full_VeReMi_SEFN_inputs"
        base = out.loc[part & out["variant"].eq(base_name)]
        if base.empty:
            continue
        b = base.iloc[0]
        for m in [
            "auroc", "auprc", "brier", "ece",
            "balanced_accuracy", "mcc", "f1", "fsr", "fur"
        ]:
            out.loc[part, f"delta_{m}_vs_full"] = (
                out.loc[part, f"{m}_mean"] - b[f"{m}_mean"]
            )
    return out

def fmt(mean, sd, digits=4):
    if pd.isna(mean):
        return "N/A"
    if pd.isna(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"

def publication_table(summary):
    rows = []
    for _, r in summary.iterrows():
        rows.append({
            "Dataset": r["dataset"],
            "Configuration": r["variant"],
            "Runs": int(r["runs"]),
            "AUROC": fmt(r["auroc_mean"], r["auroc_std"]),
            "AUPRC": fmt(r["auprc_mean"], r["auprc_std"]),
            "Brier": fmt(r["brier_mean"], r["brier_std"]),
            "ECE": fmt(r["ece_mean"], r["ece_std"]),
            "AECE": fmt(r["adaptive_ece_mean"], r["adaptive_ece_std"]),
            "NLL": fmt(r["nll_mean"], r["nll_std"]),
            "BAcc": fmt(r["balanced_accuracy_mean"], r["balanced_accuracy_std"]),
            "MCC": fmt(r["mcc_mean"], r["mcc_std"]),
            "F1": fmt(r["f1_mean"], r["f1_std"]),
            "FSR": fmt(r["fsr_mean"], r["fsr_std"]),
            "FUR": fmt(r["fur_mean"], r["fur_std"]),
            "Rel.": fmt(r["brier_reliability_mean"], r["brier_reliability_std"]),
            "Res.": fmt(r["brier_resolution_mean"], r["brier_resolution_std"]),
            "Uncert.": fmt(r["brier_uncertainty_mean"], r["brier_uncertainty_std"]),
        })
    return pd.DataFrame(rows)

def print_best(summary):
    print("\n" + "=" * 92)
    print("BEST RESULTS FROM THE 20%-TRAIN / 10%-TEST REVIEWER EXPERIMENT")
    print("=" * 92)

    for dataset in ["DABRA", "VeReMi"]:
        p = summary.loc[summary["dataset"].eq(dataset)].copy()
        if p.empty:
            continue

        # Primary ranking: AUROC, then AUPRC, then lower Brier.
        p = p.sort_values(
            ["auroc_mean", "auprc_mean", "brier_mean"],
            ascending=[False, False, True],
        )
        best = p.iloc[0]

        print(f"\n{dataset}")
        print(f"  Best configuration : {best['variant']}")
        print(f"  AUROC             : {best['auroc_mean']:.4f} ± {best['auroc_std']:.4f}")
        print(f"  AUPRC             : {best['auprc_mean']:.4f} ± {best['auprc_std']:.4f}")
        print(f"  Brier             : {best['brier_mean']:.4f} ± {best['brier_std']:.4f}")
        print(f"  ECE               : {best['ece_mean']:.4f} ± {best['ece_std']:.4f}")
        print(f"  Balanced Accuracy : {best['balanced_accuracy_mean']:.4f} ± {best['balanced_accuracy_std']:.4f}")
        print(f"  MCC               : {best['mcc_mean']:.4f} ± {best['mcc_std']:.4f}")
        print(f"  F1                : {best['f1_mean']:.4f} ± {best['f1_std']:.4f}")
        print(f"  FSR / FUR         : {best['fsr_mean']:.4f} / {best['fur_mean']:.4f}")
        print(f"  Test N            : {int(best['test_n'])}")

# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 92)
    print("REVIEWER COMPONENT / BRANCH EVALUATION")
    
    print("Seeds:", SEEDS)
    print("=" * 92)

    all_rows = []
    manifest = []

    # ---------------- DABRA ----------------
    dabra, y_d = load_dabra()
    print(f"[DABRA] rows={len(dabra):,}, positive prevalence={y_d.mean():.6f}")

    for seed in SEEDS:
        train_idx, test_idx = stratified_20_10_indices(y_d, seed)
        manifest.append({
            "dataset": "DABRA", "seed": seed,
            "total_n": len(y_d),
            "train_selected_n": len(train_idx),
            "test_n": len(test_idx),
            "unused_n": len(y_d) - len(train_idx) - len(test_idx),
        })
        for variant, groups in dabra_variants().items():
            print(f"[DABRA] seed={seed:2d}  {variant}")
            all_rows.append(
                evaluate_variant(
                    "DABRA", dabra, y_d, DABRA_FEATURES,
                    variant, groups, seed
                )
            )

    # Free memory before 1.18 GB VeReMi load.
    del dabra, y_d

    # ---------------- VeReMi ----------------
    veremi, y_v = load_veremi()
    print(f"[VeReMi] rows={len(veremi):,}, attack prevalence={y_v.mean():.6f}")

    for seed in SEEDS:
        train_idx, test_idx = stratified_20_10_indices(y_v, seed)
        manifest.append({
            "dataset": "VeReMi", "seed": seed,
            "total_n": len(y_v),
            "train_selected_n": len(train_idx),
            "test_n": len(test_idx),
            "unused_n": len(y_v) - len(train_idx) - len(test_idx),
        })
        for variant, groups in veremi_variants().items():
            print(f"[VeReMi] seed={seed:2d}  {variant}")
            all_rows.append(
                evaluate_variant(
                    "VeReMi", veremi, y_v, VEREMI_FEATURES,
                    variant, groups, seed
                )
            )

    raw = pd.DataFrame(all_rows)
    raw.to_csv(OUT / "all_seed_metrics.csv", index=False)

    pd.DataFrame(manifest).to_csv(
        OUT / "split_manifest.csv", index=False
    )

    summary = summarize(raw)
    summary = add_deltas(summary)
    summary.to_csv(OUT / "summary_metrics.csv", index=False)

    pub = publication_table(summary)
    pub.to_csv(OUT / "reviewer_table.csv", index=False)

    # LaTeX export. Wide by design; use table* + resizebox in manuscript if desired.
    latex = pub.to_latex(
        index=False,
        escape=True,
        caption=(
            "Controlled 20\\%-training/10\\%-test component and feature-branch "
            "evaluation. Values are mean $\\pm$ standard deviation over five seeds."
        ),
        label="tab:reviewer_component_20_10",
    )
    (OUT / "reviewer_table.tex").write_text(latex, encoding="utf-8")

    # Best results.
    best_rows = []
    for dataset in ["DABRA", "VeReMi"]:
        p = summary.loc[summary["dataset"].eq(dataset)].copy()
        if p.empty:
            continue
        p = p.sort_values(
            ["auroc_mean", "auprc_mean", "brier_mean"],
            ascending=[False, False, True],
        )
        best_rows.append(p.iloc[0])
    pd.DataFrame(best_rows).to_csv(
        OUT / "best_results.csv", index=False
    )

    print_best(summary)

    print("\n" + "=" * 92)
    print("PUBLICATION TABLE")
    print("=" * 92)
    print(pub.to_string(index=False))

    print("\nSaved to:")
    print(OUT)
    print("\nScientific note:")
    print(
        "VeReMi evaluates the supported V2X fault/misbehavior task. "
        "DABRA and VeReMi remain independent. SCTM is post-hoc, so this "
        "script does not fabricate a predictive 'Without_SCTM' AUROC."
    )

if __name__ == "__main__":
    main()
