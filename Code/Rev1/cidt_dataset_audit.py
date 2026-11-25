import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder


TARGET_PRIORITY = [
    "label",
    "class",
    "target",
    "attack",
    "attack_type",
    "attacktype",
    "misbehavior",
    "misbehaviour",
    "malicious",
    "is_malicious",
    "is_attack",
    "fault",
    "fault_label",
    "anomaly",
    "anomalous_event",
    "route_anomaly",
    "failure",
    "outcome",
    "y"
]

GROUP_TOKENS = [
    "driver",
    "vehicle",
    "sender",
    "receiver",
    "scenario",
    "simulation",
    "run",
    "density",
    "weather",
    "road",
    "traffic",
    "attack_type",
    "attacker",
    "trip",
    "route"
]

TIME_TOKENS = [
    "time",
    "timestamp",
    "datetime",
    "date",
    "step"
]

ID_TOKENS = [
    "id",
    "uuid",
    "index",
    "sender",
    "receiver",
    "vehicle",
    "driver",
    "trip",
    "message"
]


def snake(value):
    value = str(value).strip()
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_").lower()


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def file_sha256(path, block=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            data = handle.read(block)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def detect_format(path):
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    separators = [",", ";", "\t", "|"]
    best = None
    for encoding in encodings:
        for separator in separators:
            try:
                frame = pd.read_csv(path, nrows=200, sep=separator, encoding=encoding, low_memory=False)
                score = frame.shape[1]
                if best is None or score > best["score"]:
                    best = {
                        "encoding": encoding,
                        "separator": separator,
                        "score": score
                    }
            except Exception:
                pass
    if best is None or best["score"] < 2:
        raise RuntimeError(f"Could not detect CSV format for {path}")
    return best["encoding"], best["separator"]


def read_preview(path, encoding, separator, rows):
    frame = pd.read_csv(
        path,
        nrows=rows,
        sep=separator,
        encoding=encoding,
        low_memory=False
    )
    frame.columns = [snake(col) for col in frame.columns]
    return frame


def add_dabra_targets(frame):
    required = {
        "anomalous_event",
        "route_anomaly",
        "geofencing_violation",
        "stop_events"
    }
    if required.issubset(frame.columns):
        for tau in [2, 3, 4, 5]:
            name = f"derived_failure_tau_{tau}"
            frame[name] = (
                pd.to_numeric(frame["anomalous_event"], errors="coerce").fillna(0).astype(int).eq(1)
                | pd.to_numeric(frame["route_anomaly"], errors="coerce").fillna(0).astype(int).eq(1)
                | pd.to_numeric(frame["geofencing_violation"], errors="coerce").fillna(0).astype(int).eq(1)
                | pd.to_numeric(frame["stop_events"], errors="coerce").fillna(0).ge(tau)
            ).astype(int)
    return frame


def target_candidates(frame):
    rows = []
    for column in frame.columns:
        series = frame[column]
        unique = int(series.nunique(dropna=True))
        lname = column.lower()
        score = 0
        exact_rank = None
        if lname in TARGET_PRIORITY:
            exact_rank = TARGET_PRIORITY.index(lname)
            score += 1000 - exact_rank
        for token in TARGET_PRIORITY:
            if token in lname:
                score += 25
        if 2 <= unique <= 50:
            score += 30
        if unique == 2:
            score += 50
        if lname.startswith("derived_failure_tau_"):
            score += 2000
        if score > 0:
            rows.append({
                "column": column,
                "unique_values_preview": unique,
                "score": score,
                "dtype_preview": str(series.dtype)
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["score", "column"], ascending=[False, True]).reset_index(drop=True)


def choose_target(frame, requested):
    if requested:
        requested = snake(requested)
        if requested not in frame.columns:
            raise ValueError(f"Requested target '{requested}' not found")
        return requested
    candidates = target_candidates(frame)
    if candidates.empty:
        return None
    derived = candidates[candidates["column"].eq("derived_failure_tau_3")]
    if not derived.empty:
        return "derived_failure_tau_3"
    return str(candidates.iloc[0]["column"])


def infer_role(column):
    lname = column.lower()
    if any(token in lname for token in TIME_TOKENS):
        return "time"
    if any(token in lname for token in GROUP_TOKENS):
        return "group"
    if any(lname == token or lname.endswith(f"_{token}") for token in ID_TOKENS):
        return "identifier"
    return "feature"


def profile_preview(frame):
    rows = []
    for column in frame.columns:
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_ratio = float(numeric.notna().mean())
        top = series.astype("string").value_counts(dropna=False).head(10)
        row = {
            "column": column,
            "dtype_preview": str(series.dtype),
            "role_guess": infer_role(column),
            "preview_non_null": int(series.notna().sum()),
            "preview_missing_rate": float(series.isna().mean()),
            "preview_unique": int(series.nunique(dropna=True)),
            "numeric_parse_rate": numeric_ratio,
            "preview_mean": float(numeric.mean()) if numeric_ratio > 0.8 else np.nan,
            "preview_std": float(numeric.std()) if numeric_ratio > 0.8 else np.nan,
            "preview_min": float(numeric.min()) if numeric_ratio > 0.8 else np.nan,
            "preview_max": float(numeric.max()) if numeric_ratio > 0.8 else np.nan,
            "top_values": json.dumps({str(k): int(v) for k, v in top.items()}, ensure_ascii=False)
        }
        rows.append(row)
    return pd.DataFrame(rows)


def estimate_row_count(path, encoding, separator, chunksize):
    total = 0
    for chunk in pd.read_csv(
        path,
        sep=separator,
        encoding=encoding,
        chunksize=chunksize,
        low_memory=False
    ):
        total += len(chunk)
    return total


def full_target_counts(path, encoding, separator, target, dataset_name, chunksize):
    counts = Counter()
    total = 0
    for chunk in pd.read_csv(
        path,
        sep=separator,
        encoding=encoding,
        chunksize=chunksize,
        low_memory=False
    ):
        chunk.columns = [snake(col) for col in chunk.columns]
        if dataset_name == "dabra":
            chunk = add_dabra_targets(chunk)
        if target not in chunk.columns:
            raise ValueError(f"Target '{target}' is unavailable during full scan")
        values = chunk[target].astype("string").fillna("<NA>")
        counts.update(values.value_counts(dropna=False).to_dict())
        total += len(chunk)
    return total, counts


def balanced_plan(counts, fraction):
    valid = {str(k): int(v) for k, v in counts.items() if str(k) != "<NA>"}
    if len(valid) < 2:
        return {
            "possible": False,
            "reason": "Target has fewer than two non-missing classes",
            "fraction_requested": fraction
        }
    total = sum(valid.values())
    classes = len(valid)
    target_total = int(math.floor(total * fraction))
    per_class = min(min(valid.values()), target_total // classes)
    selected = {label: int(per_class) for label in sorted(valid)}
    actual_total = sum(selected.values())
    return {
        "possible": actual_total > 0,
        "fraction_requested": fraction,
        "source_total_non_missing": total,
        "requested_total": target_total,
        "classes": classes,
        "per_class": per_class,
        "selected_total": actual_total,
        "selected_fraction_of_source": actual_total / total if total else 0,
        "selected_per_class": selected
    }


def stratified_preview(frame, target, maximum, seed):
    if target is None or target not in frame.columns:
        return frame.sample(min(maximum, len(frame)), random_state=seed) if len(frame) > maximum else frame.copy()
    groups = []
    values = frame[target].astype("string").fillna("<NA>")
    classes = [v for v in values.unique().tolist() if v != "<NA>"]
    if len(classes) < 2:
        return frame.sample(min(maximum, len(frame)), random_state=seed) if len(frame) > maximum else frame.copy()
    each = max(1, maximum // len(classes))
    for value in classes:
        part = frame.loc[values.eq(value)]
        groups.append(part.sample(min(each, len(part)), random_state=seed))
    result = pd.concat(groups, ignore_index=True)
    if len(result) > maximum:
        result = result.sample(maximum, random_state=seed)
    return result.reset_index(drop=True)


def leakage_scan(frame, target):
    rows = []
    if target is None or target not in frame.columns:
        return pd.DataFrame(rows)
    work = frame.dropna(subset=[target]).copy()
    if work.empty:
        return pd.DataFrame(rows)
    y = work[target].astype("string")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    for column in work.columns:
        if column == target:
            continue
        series = work[column]
        unique = int(series.nunique(dropna=True))
        deterministic = False
        contradiction_groups = np.nan
        if 1 < unique <= 5000:
            grouped = work.groupby(column, dropna=False)[target].nunique(dropna=False)
            contradiction_groups = int((grouped > 1).sum())
            deterministic = contradiction_groups == 0
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            valid = numeric.notna()
            if valid.sum() > 20 and np.nanstd(numeric[valid]) > 0:
                corr = float(np.corrcoef(numeric[valid], y_encoded[valid])[0, 1])
            else:
                corr = np.nan
            bins = pd.qcut(numeric.rank(method="first"), q=min(20, max(2, int(np.sqrt(valid.sum())))), duplicates="drop")
            mi = normalized_mutual_info_score(y[valid], bins[valid].astype("string")) if valid.sum() > 20 else np.nan
        else:
            corr = np.nan
            values = series.astype("string").fillna("<NA>")
            mi = normalized_mutual_info_score(y, values)
        score = 0
        if deterministic:
            score += 100
        if not np.isnan(mi):
            score += 100 * mi
        if not np.isnan(corr):
            score += 50 * abs(corr)
        rows.append({
            "feature": column,
            "unique_values": unique,
            "deterministic_mapping_to_target": deterministic,
            "contradictory_feature_groups": contradiction_groups,
            "normalized_mutual_information": mi,
            "numeric_target_correlation": corr,
            "leakage_risk_score": score
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("leakage_risk_score", ascending=False).reset_index(drop=True)


def group_summary(frame, target):
    rows = []
    for column in frame.columns:
        if infer_role(column) != "group":
            continue
        unique = frame[column].nunique(dropna=True)
        if unique < 2 or unique > 500:
            continue
        grouped = frame.groupby(column, dropna=False).size().rename("rows").reset_index()
        if target and target in frame.columns:
            numeric_target = pd.to_numeric(frame[target], errors="coerce")
            if numeric_target.notna().mean() > 0.9:
                rates = frame.assign(_target_numeric=numeric_target).groupby(column, dropna=False)["_target_numeric"].mean()
                grouped["target_rate"] = grouped[column].map(rates)
        grouped.insert(0, "group_column", column)
        grouped = grouped.rename(columns={column: "group_value"})
        rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def timestamp_summary(frame):
    rows = []
    for column in frame.columns:
        if infer_role(column) != "time":
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce")
        rows.append({
            "column": column,
            "parse_rate": float(parsed.notna().mean()),
            "minimum": parsed.min().isoformat() if parsed.notna().any() else None,
            "maximum": parsed.max().isoformat() if parsed.notna().any() else None,
            "monotonic_increasing": bool(parsed.dropna().is_monotonic_increasing),
            "unique_values": int(parsed.nunique(dropna=True))
        })
    return pd.DataFrame(rows)


def duplicate_summary(frame):
    exact = int(frame.duplicated().sum())
    identifier_columns = [c for c in frame.columns if infer_role(c) == "identifier"]
    rows = [{
        "scope": "all_columns_preview",
        "duplicate_rows": exact,
        "duplicate_rate": exact / len(frame) if len(frame) else 0
    }]
    for column in identifier_columns:
        duplicates = int(frame[column].duplicated(keep=False).sum())
        rows.append({
            "scope": column,
            "duplicate_rows": duplicates,
            "duplicate_rate": duplicates / len(frame) if len(frame) else 0
        })
    return pd.DataFrame(rows)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=json_safe)


def analyze_dataset(name, path, output_dir, requested_target, preview_rows, chunksize, fraction, seed):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    encoding, separator = detect_format(path)
    preview = read_preview(path, encoding, separator, preview_rows)
    if name == "dabra":
        preview = add_dabra_targets(preview)
    candidates = target_candidates(preview)
    target = choose_target(preview, requested_target)
    total_rows, counts = full_target_counts(path, encoding, separator, target, name, chunksize) if target else (estimate_row_count(path, encoding, separator, chunksize), Counter())
    plan = balanced_plan(counts, fraction) if target else {
        "possible": False,
        "reason": "No target detected",
        "fraction_requested": fraction
    }
    stratified = stratified_preview(preview, target, min(200000, preview_rows), seed)
    profiles = profile_preview(preview)
    leakage = leakage_scan(stratified, target)
    groups = group_summary(stratified, target)
    times = timestamp_summary(preview)
    duplicates = duplicate_summary(stratified)
    candidates.to_csv(dataset_dir / "target_candidates.csv", index=False)
    profiles.to_csv(dataset_dir / "column_profile.csv", index=False)
    leakage.to_csv(dataset_dir / "leakage_scan.csv", index=False)
    groups.to_csv(dataset_dir / "group_summary.csv", index=False)
    times.to_csv(dataset_dir / "timestamp_summary.csv", index=False)
    duplicates.to_csv(dataset_dir / "duplicate_summary.csv", index=False)
    stratified.to_csv(dataset_dir / "analysis_sample.csv", index=False)
    class_rows = [{"class": str(k), "count": int(v), "rate": int(v) / total_rows if total_rows else 0} for k, v in counts.items()]
    pd.DataFrame(class_rows).sort_values("count", ascending=False).to_csv(dataset_dir / "class_distribution.csv", index=False)
    overview = {
        "dataset": name,
        "path": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "encoding": encoding,
        "separator": separator,
        "total_rows": total_rows,
        "raw_columns": int(len(read_preview(path, encoding, separator, 1).columns)),
        "analysis_columns": int(len(preview.columns)),
        "selected_target": target,
        "target_counts": {str(k): int(v) for k, v in counts.items()},
        "balanced_50_plan": plan,
        "preview_rows_loaded": int(len(preview)),
        "analysis_sample_rows": int(len(stratified)),
        "candidate_group_columns": [c for c in preview.columns if infer_role(c) == "group"],
        "candidate_time_columns": [c for c in preview.columns if infer_role(c) == "time"],
        "candidate_identifier_columns": [c for c in preview.columns if infer_role(c) == "identifier"],
        "high_leakage_features": leakage.loc[leakage["leakage_risk_score"].ge(80), "feature"].tolist() if not leakage.empty else []
    }
    save_json(dataset_dir / "overview.json", overview)
    return overview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dabra", default=r"D:\other\Causal-DT\Dataset\DABRA.csv")
    parser.add_argument("--veremi", default=r"D:\other\Causal-DT\Dataset\VeReMi.csv")
    parser.add_argument("--out", default=r"D:\other\Causal-DT\DatasetAudit")
    parser.add_argument("--dabra-target", default=None)
    parser.add_argument("--veremi-target", default=None)
    parser.add_argument("--preview-rows", type=int, default=200000)
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument("--fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    failures = {}
    for name, path, target in [
        ("dabra", args.dabra, args.dabra_target),
        ("veremi", args.veremi, args.veremi_target)
    ]:
        try:
            results[name] = analyze_dataset(
                name=name,
                path=path,
                output_dir=output_dir,
                requested_target=target,
                preview_rows=args.preview_rows,
                chunksize=args.chunksize,
                fraction=args.fraction,
                seed=args.seed
            )
            print(f"{name}: completed")
            print(json.dumps(results[name], indent=2, default=json_safe))
        except Exception as error:
            failures[name] = {
                "type": type(error).__name__,
                "message": str(error)
            }
            print(f"{name}: failed: {error}", file=sys.stderr)
    save_json(output_dir / "combined_overview.json", {
        "completed": results,
        "failures": failures
    })
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
