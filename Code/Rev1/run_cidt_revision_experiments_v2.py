import argparse
import gc
import json
import math
import os
import pickle
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
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
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


@dataclass
class RunConfig:
    profile: str
    fraction: float
    seeds: list
    epochs: int
    patience: int
    batch_size: int
    bootstrap: int
    veremi_limit: int
    workers: int
    device: str


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snake(value):
    import re
    value = str(value).strip()
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_").lower()


def safe_json(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=safe_json)


def logit_clip(prob):
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(prob / (1 - prob))


def sigmoid_np(value):
    value = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40, 40)))


def expected_calibration_error(y, p, bins=15):
    y = np.asarray(y)
    p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    score = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & (p < right if right < 1 else p <= right)
        if mask.any():
            score += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(score)


def adaptive_calibration_error(y, p, bins=15):
    y = np.asarray(y)
    p = np.asarray(p)
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    score = 0.0
    for chunk in chunks:
        if len(chunk):
            score += len(chunk) / len(y) * abs(y[chunk].mean() - p[chunk].mean())
    return float(score)


def brier_decomposition(y, p, bins=15):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    base = y.mean()
    reliability = 0.0
    resolution = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & (p < right if right < 1 else p <= right)
        if mask.any():
            weight = mask.mean()
            obs = y[mask].mean()
            pred = p[mask].mean()
            reliability += weight * (pred - obs) ** 2
            resolution += weight * (obs - base) ** 2
    uncertainty = base * (1 - base)
    return float(reliability), float(resolution), float(uncertainty)


def choose_threshold(y, p, max_fsr=0.05):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    candidates = np.unique(np.concatenate([np.linspace(0.001, 0.999, 500), p]))
    best = None
    for threshold in candidates:
        pred = (p >= threshold).astype(int)
        positives = y == 1
        negatives = y == 0
        fsr = float(((pred == 0) & positives).sum() / max(1, positives.sum()))
        fur = float(((pred == 1) & negatives).sum() / max(1, negatives.sum()))
        bacc = balanced_accuracy_score(y, pred)
        valid = fsr <= max_fsr
        key = (int(valid), bacc, -fur, -threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def calibration_fit(y, p):
    model = LogisticRegression(solver="lbfgs")
    model.fit(logit_clip(p).reshape(-1, 1), np.asarray(y, dtype=int))
    return model


def calibration_apply(model, p):
    return model.predict_proba(logit_clip(p).reshape(-1, 1))[:, 1]


def compute_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    pred = (p >= threshold).astype(int)
    pos = y == 1
    neg = y == 0
    fsr = float(((pred == 0) & pos).sum() / max(1, pos.sum()))
    fur = float(((pred == 1) & neg).sum() / max(1, neg.sum()))
    reliability, resolution, uncertainty = brier_decomposition(y, p)
    result = {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)) if np.unique(y).size > 1 else np.nan,
        "auprc": float(average_precision_score(y, p)) if np.unique(y).size > 1 else np.nan,
        "auprc_gain": float(average_precision_score(y, p) - y.mean()) if np.unique(y).size > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "adaptive_ece": adaptive_calibration_error(y, p),
        "nll": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fsr": fsr,
        "fur": fur,
        "threshold": float(threshold),
        "brier_reliability": reliability,
        "brier_resolution": resolution,
        "brier_uncertainty": uncertainty,
    }
    return result


def bootstrap_ci(y, p, metric, repetitions, seed, maximum=50000):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    if len(y) > maximum:
        idx = rng.choice(len(y), maximum, replace=False)
        y = y[idx]
        p = p[idx]
    values = []
    for _ in range(repetitions):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        pb = p[idx]
        if np.unique(yb).size < 2:
            continue
        if metric == "auroc":
            values.append(roc_auc_score(yb, pb))
        elif metric == "auprc":
            values.append(average_precision_score(yb, pb))
        elif metric == "brier":
            values.append(brier_score_loss(yb, pb))
    if not values:
        return np.nan, np.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def balanced_binary_sample(frame, target, n_total, seed):
    n_each = n_total // 2
    parts = []
    for value in [0, 1]:
        group = frame.loc[frame[target].eq(value)]
        if len(group) < n_each:
            raise ValueError(f"Class {value} has {len(group)} rows, below required {n_each}")
        parts.append(group.sample(n=n_each, random_state=seed))
    result = pd.concat(parts, ignore_index=True)
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def allocate_attack_quotas(counts, target):
    attack_counts = counts.drop(index=0, errors="ignore")
    if attack_counts.empty:
        raise ValueError("No VeReMi attack classes were found after excluding class 0")
    weights = attack_counts / attack_counts.sum()
    quotas = np.floor(weights * target).astype(int)
    remainder = int(target - quotas.sum())
    fractions = weights * target - quotas
    for label in fractions.sort_values(ascending=False).index[:remainder]:
        quotas.loc[label] += 1
    return quotas.to_dict()


def load_dabra(path, fraction, seed, out):
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [snake(c) for c in frame.columns]
    required = ["anomalous_event", "route_anomaly", "geofencing_violation", "stop_events"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"DABRA missing columns: {missing}")
    frame["target"] = (
        frame["anomalous_event"].astype(int).eq(1)
        | frame["route_anomaly"].astype(int).eq(1)
        | frame["geofencing_violation"].astype(int).eq(1)
        | pd.to_numeric(frame["stop_events"], errors="coerce").fillna(0).ge(3)
    ).astype(np.int8)
    desired = int(math.floor(len(frame) * fraction))
    desired = desired - desired % 2
    maximum = 2 * int(frame["target"].value_counts().min())
    desired = min(desired, maximum)
    frame = balanced_binary_sample(frame, "target", desired, seed)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["environment"] = (
        frame["weather_conditions"].astype(str)
        + "|"
        + frame["road_type"].astype(str)
        + "|"
        + frame["traffic_condition"].astype(str)
    )
    frame["environment_id"] = pd.factorize(frame["environment"])[0].astype(np.int16)
    frame["row_id"] = frame["trip_id"].astype(str)
    frame.to_parquet(out / "data" / "dabra_balanced.parquet", index=False)
    return frame


def load_veremi(path, fraction, seed, out, limit):
    usecols = [
        "type", "sendtime", "sender", "senderpseudo", "messageid", "class",
        "posx", "posy", "posz", "posx_n", "posy_n", "posz_n",
        "spdx", "spdy", "spdz", "spdx_n", "spdy_n", "spdz_n",
        "aclx", "acly", "aclz", "aclx_n", "acly_n", "aclz_n",
        "hedx", "hedy", "hedz", "hedx_n", "hedy_n", "hedz_n"
    ]
    header = pd.read_csv(path, nrows=0)
    original_columns = list(header.columns)
    normalized_map = {}
    for original in original_columns:
        normalized = snake(original)
        if normalized in normalized_map and normalized_map[normalized] != original:
            raise ValueError(f"Duplicate normalized VeReMi column name: {normalized}")
        normalized_map[normalized] = original
    missing = [column for column in usecols if column not in normalized_map]
    if missing:
        raise ValueError(
            "VeReMi columns unavailable after normalization: "
            + str(missing)
            + ". Available normalized columns: "
            + str(sorted(normalized_map))
        )
    actual_usecols = [normalized_map[column] for column in usecols]
    dtypes = {normalized_map[column]: "float32" for column in usecols}
    for column in ["type", "sender", "senderpseudo", "messageid", "class"]:
        dtypes[normalized_map[column]] = "int64"
    frame = pd.read_csv(
        path,
        usecols=actual_usecols,
        dtype=dtypes,
        low_memory=False
    )
    frame.columns = [snake(column) for column in frame.columns]
    frame = frame[usecols]
    frame["target"] = frame["class"].ne(0).astype(np.int8)
    desired = int(math.floor(len(frame) * fraction))
    desired = desired - desired % 2
    if limit > 0:
        desired = min(desired, limit - limit % 2)
    n_each = desired // 2
    normal = frame.loc[frame["class"].eq(0)].sample(n=n_each, random_state=seed)
    attack_counts = frame.loc[frame["class"].ne(0), "class"].value_counts().sort_index()
    quotas = allocate_attack_quotas(attack_counts, n_each)
    attack_parts = []
    for label, quota in quotas.items():
        group = frame.loc[frame["class"].eq(label)]
        attack_parts.append(group.sample(n=int(quota), random_state=seed))
    attacks = pd.concat(attack_parts, ignore_index=True)
    frame = pd.concat([normal, attacks], ignore_index=True)
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    frame["time_bin"] = pd.qcut(frame["sendtime"], q=8, labels=False, duplicates="drop").astype(np.int16)
    frame["environment_id"] = frame["time_bin"]
    frame["row_id"] = frame["messageid"].astype(str)
    frame.to_parquet(out / "data" / "veremi_balanced.parquet", index=False)
    save_json(out / "data" / "veremi_attack_quotas.json", quotas)
    del normal, attacks, attack_parts
    gc.collect()
    return frame


def stratified_three_way(y, seed):
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(indices, test_size=0.4, stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=y[temp_idx], random_state=seed)
    return train_idx, val_idx, test_idx


def grouped_three_way(y, groups, seed):
    indices = np.arange(len(y))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_val, test = next(splitter.split(indices, y, groups))
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    train_rel, val_rel = next(splitter2.split(train_val, y[train_val], np.asarray(groups)[train_val]))
    train = train_val[train_rel]
    val = train_val[val_rel]
    return train, val, test


def temporal_three_way(values):
    order = np.argsort(np.asarray(values))
    n = len(order)
    a = int(n * 0.6)
    b = int(n * 0.8)
    return order[:a], order[a:b], order[b:]


def dabra_splits(frame, seed):
    y = frame["target"].to_numpy()
    splits = {"iid": stratified_three_way(y, seed)}
    splits["temporal"] = temporal_three_way(frame["timestamp"].astype("int64").to_numpy())
    splits["environment_heldout"] = grouped_three_way(y, frame["environment"].to_numpy(), seed)
    for driver in sorted(frame["driver_id"].unique()):
        test = np.where(frame["driver_id"].to_numpy() == driver)[0]
        remain = np.where(frame["driver_id"].to_numpy() != driver)[0]
        train, val = train_test_split(remain, test_size=0.2, stratify=y[remain], random_state=seed)
        splits[f"driver_{driver}_heldout"] = (train, val, test)
    return splits


def veremi_unseen_attack_split(frame, seed):
    rng = np.random.default_rng(seed)
    normal = np.where(frame["class"].to_numpy() == 0)[0]
    rng.shuffle(normal)
    a = int(len(normal) * 0.7)
    b = int(len(normal) * 0.85)
    normal_train, normal_val, normal_test = normal[:a], normal[a:b], normal[b:]
    classes = sorted([int(c) for c in frame["class"].unique() if int(c) != 0])
    train_classes = classes[:14]
    val_classes = classes[14:16]
    test_classes = classes[16:]
    attack_train = np.where(frame["class"].isin(train_classes).to_numpy())[0]
    attack_val = np.where(frame["class"].isin(val_classes).to_numpy())[0]
    attack_test = np.where(frame["class"].isin(test_classes).to_numpy())[0]
    train = np.concatenate([normal_train, attack_train])
    val = np.concatenate([normal_val, attack_val])
    test = np.concatenate([normal_test, attack_test])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test, {
        "train_attack_classes": train_classes,
        "validation_attack_classes": val_classes,
        "test_attack_classes": test_classes
    }


def veremi_splits(frame, seed):
    y = frame["target"].to_numpy()
    splits = {
        "iid": stratified_three_way(y, seed),
        "sender_heldout": grouped_three_way(y, frame["sender"].to_numpy(), seed),
        "temporal": temporal_three_way(frame["sendtime"].to_numpy()),
    }
    train, val, test, mapping = veremi_unseen_attack_split(frame, seed)
    splits["unseen_attack"] = (train, val, test)
    return splits, mapping


class DatasetPreprocessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.scalers = {}
        self.encoder = None
        if dataset == "dabra":
            self.branches = {
                "behavior": [
                    "latitude", "longitude", "speed", "acceleration", "steering_angle",
                    "heading", "trip_duration", "trip_distance", "fuel_consumption",
                    "rpm", "brake_usage", "lane_deviation", "route_deviation_score",
                    "acceleration_variation", "behavioral_consistency_index"
                ],
                "environment": ["weather_conditions", "road_type", "traffic_condition"]
            }
        else:
            self.branches = {
                "position": ["posx", "posy", "posx_n", "posy_n"],
                "speed": ["spdx", "spdy", "spdx_n", "spdy_n"],
                "acceleration": ["aclx", "acly", "aclx_n", "acly_n"],
                "heading": ["hedx", "hedy", "hedx_n", "hedy_n"]
            }

    def fit(self, frame):
        if self.dataset == "dabra":
            numeric = self.branches["behavior"]
            self.scalers["behavior"] = StandardScaler().fit(frame[numeric])
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
            self.encoder.fit(frame[self.branches["environment"]].astype(str))
        else:
            for name, cols in self.branches.items():
                self.scalers[name] = StandardScaler().fit(frame[cols])
        return self

    def transform(self, frame):
        arrays = []
        dims = []
        names = []
        if self.dataset == "dabra":
            behavior = self.scalers["behavior"].transform(frame[self.branches["behavior"]]).astype(np.float32)
            environment = self.encoder.transform(frame[self.branches["environment"]].astype(str)).astype(np.float32)
            arrays.extend([behavior, environment])
            dims.extend([behavior.shape[1], environment.shape[1]])
            names.extend(["behavior", "environment"])
        else:
            for name, cols in self.branches.items():
                arr = self.scalers[name].transform(frame[cols]).astype(np.float32)
                arrays.append(arr)
                dims.append(arr.shape[1])
                names.append(name)
        return np.concatenate(arrays, axis=1), dims, names


class ConstantModel:
    def __init__(self, value):
        self.value = float(value)

    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        p = np.full(len(x), self.value, dtype=float)
        return np.column_stack([1 - p, p])


class PrevalenceModel:
    def fit(self, x, y):
        self.value = float(np.mean(y))
        return self

    def predict_proba(self, x):
        p = np.full(len(x), self.value, dtype=float)
        return np.column_stack([1 - p, p])


class MultiBranchNet(nn.Module):
    def __init__(self, branch_dims, branch_names, hidden=64, dropout=0.15, disabled=None, domains=1):
        super().__init__()
        self.branch_dims = list(branch_dims)
        self.branch_names = list(branch_names)
        self.disabled = set(disabled or [])
        self.encoders = nn.ModuleList()
        for dim in branch_dims:
            self.encoders.append(nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU()
            ))
        self.fusion = nn.Sequential(
            nn.Linear(hidden * len(branch_dims), hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU()
        )
        self.head = nn.Linear(hidden, 1)
        self.domain_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, domains)
        )

    def forward(self, x):
        outputs = []
        offset = 0
        for dim, name, encoder in zip(self.branch_dims, self.branch_names, self.encoders):
            segment = x[:, offset:offset + dim]
            offset += dim
            if name in self.disabled:
                outputs.append(torch.zeros((len(x), encoder[0].out_features), device=x.device, dtype=x.dtype))
            else:
                outputs.append(encoder(segment))
        representation = self.fusion(torch.cat(outputs, dim=1))
        logits = self.head(representation).squeeze(1)
        return logits, representation


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.strength * grad_output, None


def irm_penalty(logits, y, env):
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    penalties = []
    losses = []
    for value in torch.unique(env):
        mask = env == value
        if mask.sum() < 2:
            continue
        loss = F.binary_cross_entropy_with_logits(logits[mask] * scale, y[mask])
        gradient = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        penalties.append(gradient.pow(2))
        losses.append(loss)
    if not losses:
        loss = F.binary_cross_entropy_with_logits(logits, y)
        return loss, torch.zeros((), device=logits.device)
    return torch.stack(losses).mean(), torch.stack(penalties).mean()


def covariance(value):
    value = value - value.mean(dim=0, keepdim=True)
    denominator = max(1, value.shape[0] - 1)
    return value.t().mm(value) / denominator


def coral_penalty(representation, env):
    groups = []
    for value in torch.unique(env)[:6]:
        group = representation[env == value]
        if len(group) >= 4:
            groups.append(group)
    if len(groups) < 2:
        return torch.zeros((), device=representation.device)
    total = torch.zeros((), device=representation.device)
    pairs = 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            mean = (groups[i].mean(0) - groups[j].mean(0)).pow(2).mean()
            cov = (covariance(groups[i]) - covariance(groups[j])).pow(2).mean()
            total = total + mean + cov
            pairs += 1
    return total / max(1, pairs)


def mmd_penalty(representation, env):
    means = []
    for value in torch.unique(env)[:8]:
        group = representation[env == value]
        if len(group) >= 2:
            means.append(group.mean(0))
    if len(means) < 2:
        return torch.zeros((), device=representation.device)
    stacked = torch.stack(means)
    return stacked.var(dim=0, unbiased=False).mean()


def group_risk_variance(logits, y, env):
    risks = []
    for value in torch.unique(env):
        mask = env == value
        if mask.sum() >= 2:
            risks.append(F.binary_cross_entropy_with_logits(logits[mask], y[mask]))
    if len(risks) < 2:
        return torch.zeros((), device=logits.device)
    return torch.stack(risks).var(unbiased=False)


def neural_objective(name, logits, representation, y, env, domain_head, epoch, total_epochs):
    base = F.binary_cross_entropy_with_logits(logits, y)
    extra = {}
    if name == "ERM":
        return base, extra
    if name == "IRM":
        risk, penalty = irm_penalty(logits, y, env)
        weight = 1.0 if epoch < max(2, total_epochs // 4) else 10.0
        extra["irm"] = penalty
        return risk + weight * penalty, extra
    if name == "CORAL":
        penalty = coral_penalty(representation, env)
        extra["coral"] = penalty
        return base + 0.1 * penalty, extra
    if name == "MMD":
        penalty = mmd_penalty(representation, env)
        extra["mmd"] = penalty
        return base + 0.1 * penalty, extra
    if name == "DANN":
        strength = min(1.0, epoch / max(1, total_epochs - 1))
        reversed_representation = GradientReverse.apply(representation, strength)
        domain_logits = domain_head(reversed_representation)
        domain_loss = F.cross_entropy(domain_logits, env)
        extra["domain"] = domain_loss
        return base + 0.2 * domain_loss, extra
    if name == "GroupDRO":
        risks = []
        for value in torch.unique(env):
            mask = env == value
            if mask.sum() >= 2:
                risks.append(F.binary_cross_entropy_with_logits(logits[mask], y[mask]))
        worst = torch.stack(risks).max() if risks else base
        extra["worst"] = worst
        return worst, extra
    if name == "CIDT":
        risk, irm = irm_penalty(logits, y, env)
        coral = coral_penalty(representation, env)
        variance = group_risk_variance(logits, y, env)
        weight = 1.0 if epoch < max(2, total_epochs // 4) else 10.0
        extra["irm"] = irm
        extra["coral"] = coral
        extra["variance"] = variance
        return risk + weight * irm + 0.1 * coral + 0.1 * variance, extra
    return base, extra


def predict_neural(model, x, device, batch=8192, return_representation=False):
    model.eval()
    probabilities = []
    representations = []
    tensor = torch.from_numpy(x.astype(np.float32))
    loader = DataLoader(TensorDataset(tensor), batch_size=batch, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits, representation = model(xb)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            if return_representation:
                representations.append(representation.cpu().numpy())
    p = np.concatenate(probabilities)
    if return_representation:
        return p, np.concatenate(representations)
    return p


def train_neural(name, x_train, y_train, env_train, x_val, y_val, branch_dims, branch_names, seed, config, disabled=None, irm_lambda=None):
    seed_everything(seed)
    device = torch.device(config.device)
    unique_domains = np.unique(env_train)
    domain_map = {value: index for index, value in enumerate(unique_domains)}
    mapped_env = np.array([domain_map.get(v, 0) for v in env_train], dtype=np.int64)
    model = MultiBranchNet(branch_dims, branch_names, domains=max(1, len(unique_domains)), disabled=disabled).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
        torch.from_numpy(mapped_env)
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=False)
    best_state = None
    best_score = -np.inf
    wait = 0
    history = []
    start = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for xb, yb, eb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            eb = eb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, representation = model(xb)
            objective_name = name
            loss, extras = neural_objective(objective_name, logits, representation, yb, eb, model.domain_head, epoch, config.epochs)
            if irm_lambda is not None:
                risk, penalty = irm_penalty(logits, yb, eb)
                loss = risk + float(irm_lambda) * penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_p = predict_neural(model, x_val, device)
        val_score = average_precision_score(y_val, val_p) if np.unique(y_val).size > 1 else -np.inf
        history.append({
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "validation_auprc": float(val_score)
        })
        if val_score > best_score + 1e-5:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - start
    return model, pd.DataFrame(history), elapsed


def fit_sklearn(name, x_train, y_train, seed, workers):
    if name == "Constant05":
        model = ConstantModel(0.5)
    elif name == "Prevalence":
        model = PrevalenceModel()
    elif name == "LR":
        model = LogisticRegression(
            solver="saga",
            max_iter=500,
            class_weight="balanced",
            n_jobs=workers,
            random_state=seed
        )
    elif name == "XGB":
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=workers,
            random_state=seed
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            random_state=seed
        )
    start = time.perf_counter()
    model.fit(x_train, y_train)
    elapsed = time.perf_counter() - start
    return model, elapsed


def predict_sklearn(model, x):
    return model.predict_proba(x)[:, 1]


def model_size_mb(model, path):
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), path)
    else:
        joblib.dump(model, path)
    return path.stat().st_size / (1024 * 1024)


def latency_sklearn(model, x, repeats=200):
    sample = x[:1]
    for _ in range(20):
        model.predict_proba(sample)
    values = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        model.predict_proba(sample)
        values.append((time.perf_counter_ns() - start) / 1e6)
    return float(np.median(values)), float(np.quantile(values, 0.95))


def latency_neural(model, x, device, repeats=200):
    model = model.to(device).eval()
    sample = torch.from_numpy(x[:1].astype(np.float32)).to(device)
    with torch.no_grad():
        for _ in range(20):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        values = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            with torch.no_grad():
                model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            values.append((time.perf_counter_ns() - start) / 1e6)
    return float(np.median(values)), float(np.quantile(values, 0.95))


def write_predictions(path, frame, indices, y, p, threshold, dataset, split, model, seed, calibration):
    path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame({
        "row_id": frame.iloc[indices]["row_id"].astype(str).to_numpy(),
        "dataset": dataset,
        "split": split,
        "model": model,
        "seed": seed,
        "calibration": calibration,
        "true_label": y.astype(np.int8),
        "predicted_probability": p.astype(np.float32),
        "predicted_class": (p >= threshold).astype(np.int8),
        "threshold": np.float32(threshold)
    })
    if dataset == "dabra":
        result["group_id"] = frame.iloc[indices]["environment"].astype(str).to_numpy()
    else:
        result["group_id"] = frame.iloc[indices]["sender"].astype(str).to_numpy()
        result["attack_class"] = frame.iloc[indices]["class"].astype(np.int16).to_numpy()
    result.to_parquet(path, index=False)


def append_metric(rows, dataset, split, model, seed, partition, calibration, metrics, train_time, infer_median, infer_p95, size_mb, disabled):
    row = {
        "dataset": dataset,
        "split": split,
        "model": model,
        "seed": seed,
        "partition": partition,
        "calibration": calibration,
        "training_seconds": train_time,
        "latency_median_ms": infer_median,
        "latency_p95_ms": infer_p95,
        "model_size_mb": size_mb,
        "disabled_branches": "|".join(disabled or [])
    }
    row.update(metrics)
    rows.append(row)


def run_single_model(dataset, frame, split_name, indices, preprocessor, x_all, branch_dims, branch_names, model_name, seed, config, out, disabled=None, irm_lambda=None, run_label=None):
    label = run_label or model_name
    train_idx, val_idx, test_idx = indices
    y_all = frame["target"].to_numpy(dtype=np.int8)
    env_all = frame["environment_id"].to_numpy(dtype=np.int64)
    x_train, y_train, env_train = x_all[train_idx], y_all[train_idx], env_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]
    x_test, y_test = x_all[test_idx], y_all[test_idx]
    history = pd.DataFrame()
    if model_name in ["Constant05", "Prevalence", "LR", "XGB", "HGB"]:
        model, training_seconds = fit_sklearn(model_name, x_train, y_train, seed, config.workers)
        train_p = predict_sklearn(model, x_train)
        val_p = predict_sklearn(model, x_val)
        test_p = predict_sklearn(model, x_test)
        infer_median, infer_p95 = latency_sklearn(model, x_test)
        model_path = out / "models" / dataset / split_name / f"{label}_seed{seed}.joblib"
    else:
        model, history, training_seconds = train_neural(
            model_name,
            x_train,
            y_train,
            env_train,
            x_val,
            y_val,
            branch_dims,
            branch_names,
            seed,
            config,
            disabled=disabled,
            irm_lambda=irm_lambda
        )
        train_p = predict_neural(model, x_train, torch.device(config.device))
        val_p = predict_neural(model, x_val, torch.device(config.device))
        test_p = predict_neural(model, x_test, torch.device(config.device))
        infer_median, infer_p95 = latency_neural(model, x_test, torch.device(config.device))
        model_path = out / "models" / dataset / split_name / f"{label}_seed{seed}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    size_mb = model_size_mb(model, model_path)
    threshold_raw = choose_threshold(y_val, val_p)
    calibrator = calibration_fit(y_val, val_p)
    train_pc = calibration_apply(calibrator, train_p)
    val_pc = calibration_apply(calibrator, val_p)
    test_pc = calibration_apply(calibrator, test_p)
    threshold_cal = choose_threshold(y_val, val_pc)
    rows = []
    for partition, y, p, threshold in [
        ("train", y_train, train_p, threshold_raw),
        ("validation", y_val, val_p, threshold_raw),
        ("test", y_test, test_p, threshold_raw)
    ]:
        append_metric(rows, dataset, split_name, label, seed, partition, "raw", compute_metrics(y, p, threshold), training_seconds, infer_median, infer_p95, size_mb, disabled)
    for partition, y, p, threshold in [
        ("train", y_train, train_pc, threshold_cal),
        ("validation", y_val, val_pc, threshold_cal),
        ("test", y_test, test_pc, threshold_cal)
    ]:
        append_metric(rows, dataset, split_name, label, seed, partition, "platt", compute_metrics(y, p, threshold), training_seconds, infer_median, infer_p95, size_mb, disabled)
    for metric in ["auroc", "auprc", "brier"]:
        low, high = bootstrap_ci(y_test, test_pc, metric, config.bootstrap, seed)
        for row in rows:
            if row["partition"] == "test" and row["calibration"] == "platt":
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
    pred_dir = out / "predictions" / dataset / split_name
    if model_name in ["Constant05", "Prevalence", "XGB", "ERM", "IRM", "CIDT"]:
        write_predictions(pred_dir / f"{label}_seed{seed}_raw.parquet", frame, test_idx, y_test, test_p, threshold_raw, dataset, split_name, label, seed, "raw")
        write_predictions(pred_dir / f"{label}_seed{seed}_platt.parquet", frame, test_idx, y_test, test_pc, threshold_cal, dataset, split_name, label, seed, "platt")
    if not history.empty:
        history["dataset"] = dataset
        history["split"] = split_name
        history["model"] = label
        history["seed"] = seed
        history.to_csv(out / "history" / f"{dataset}_{split_name}_{label}_seed{seed}.csv", index=False)
    joblib.dump(calibrator, out / "models" / dataset / split_name / f"{label}_seed{seed}_calibrator.joblib")
    return rows, model, (x_test, y_test, test_idx, test_p, test_pc, threshold_raw, threshold_cal)


def quantized_evaluation(dataset, frame, split_name, seed, model, package, config, out):
    x_test, y_test, test_idx, raw_p, calibrated_p, threshold_raw, threshold_cal = package
    if not isinstance(model, nn.Module):
        return []
    cpu_model = model.to("cpu").eval()
    try:
        quantized = torch.ao.quantization.quantize_dynamic(cpu_model, {nn.Linear}, dtype=torch.qint8)
    except Exception:
        return []
    p = predict_neural(quantized, x_test, torch.device("cpu"))
    threshold = threshold_raw
    median, p95 = latency_neural(quantized, x_test, torch.device("cpu"))
    path = out / "models" / dataset / split_name / f"CIDT_INT8_seed{seed}.pt"
    size = model_size_mb(quantized, path)
    metrics = compute_metrics(y_test, p, threshold)
    row = {
        "dataset": dataset,
        "split": split_name,
        "model": "CIDT_INT8",
        "seed": seed,
        "partition": "test",
        "calibration": "raw",
        "training_seconds": 0.0,
        "latency_median_ms": median,
        "latency_p95_ms": p95,
        "model_size_mb": size,
        "disabled_branches": ""
    }
    row.update(metrics)
    write_predictions(
        out / "predictions" / dataset / split_name / f"CIDT_INT8_seed{seed}.parquet",
        frame,
        test_idx,
        y_test,
        p,
        threshold,
        dataset,
        split_name,
        "CIDT_INT8",
        seed,
        "raw"
    )
    return [row]


def select_models_for_split(dataset, split_name):
    all_models = ["Constant05", "Prevalence", "LR", "XGB", "ERM", "IRM", "DANN", "GroupDRO", "CORAL", "MMD", "CIDT"]
    principal = ["Constant05", "Prevalence", "XGB", "ERM", "IRM", "GroupDRO", "CIDT"]
    compact = ["Constant05", "Prevalence", "ERM", "IRM", "CIDT"]
    if split_name == "iid":
        return all_models
    if split_name in ["environment_heldout", "sender_heldout", "unseen_attack"]:
        return principal
    return compact


def ablation_spec(dataset, branch_names):
    specs = [("CIDT_no_IRM", [], 0.0)]
    for branch in branch_names:
        specs.append((f"CIDT_without_{branch}", [branch], None))
    return specs


def run_dataset(dataset, frame, splits, config, out):
    all_rows = []
    first_seed = config.seeds[0]
    for split_name, indices in splits.items():
        train_idx, val_idx, test_idx = indices
        preprocessor = DatasetPreprocessor(dataset).fit(frame.iloc[train_idx])
        x_all, branch_dims, branch_names = preprocessor.transform(frame)
        split_dir = out / "preprocessors" / dataset / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, split_dir / "preprocessor.joblib")
        models = select_models_for_split(dataset, split_name)
        for model_name in models:
            seeds = config.seeds if model_name in ["ERM", "IRM", "DANN", "GroupDRO", "CORAL", "MMD", "CIDT"] else [first_seed]
            for seed in seeds:
                rows, model, package = run_single_model(
                    dataset, frame, split_name, indices, preprocessor, x_all, branch_dims,
                    branch_names, model_name, seed, config, out
                )
                all_rows.extend(rows)
                if model_name == "CIDT" and seed == first_seed and split_name in ["iid", "environment_heldout", "unseen_attack"]:
                    all_rows.extend(quantized_evaluation(dataset, frame, split_name, seed, model, package, config, out))
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        key_split = "environment_heldout" if dataset == "dabra" else "unseen_attack"
        if split_name == key_split:
            for ablation_name, disabled, irm_lambda in ablation_spec(dataset, branch_names):
                rows, model, package = run_single_model(
                    dataset, frame, split_name, indices, preprocessor, x_all, branch_dims,
                    branch_names, "CIDT", first_seed, config, out, disabled=disabled, irm_lambda=irm_lambda, run_label=ablation_name
                )
                all_rows.extend(rows)
                del model
                gc.collect()
            for lam in [0.0, 0.1, 1.0, 10.0, 100.0]:
                rows, model, package = run_single_model(
                    dataset, frame, split_name, indices, preprocessor, x_all, branch_dims,
                    branch_names, "IRM", first_seed, config, out, irm_lambda=lam, run_label=f"IRM_lambda_{lam:g}"
                )
                all_rows.extend(rows)
                del model
                gc.collect()
        del x_all
        gc.collect()
    return pd.DataFrame(all_rows)


def calibration_points(y, p, bins=10):
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), q=bins, labels=False, duplicates="drop")
    result = frame.groupby("bin", as_index=False).agg(
        mean_predicted=("p", "mean"),
        observed_rate=("y", "mean"),
        count=("y", "size")
    )
    return result


def create_calibration_figures(out):
    source = out / "figure_sources"
    figures = out / "figures"
    rows = []
    for dataset in ["dabra", "veremi"]:
        folder = out / "predictions" / dataset / "iid"
        if not folder.exists():
            continue
        files = sorted(folder.glob("*_seed11_platt.parquet"))
        for file in files:
            model = file.name.split("_seed11_")[0]
            if model not in ["Constant05", "Prevalence", "XGB", "ERM", "IRM", "CIDT"]:
                continue
            frame = pd.read_parquet(file, columns=["true_label", "predicted_probability"])
            points = calibration_points(frame["true_label"], frame["predicted_probability"])
            points["dataset"] = dataset
            points["model"] = model
            rows.append(points)
        dataset_rows = [r for r in rows if not r.empty and r["dataset"].iloc[0] == dataset]
        if dataset_rows:
            combined = pd.concat(dataset_rows, ignore_index=True)
            plt.figure(figsize=(5.5, 4.2))
            plt.plot([0, 1], [0, 1], linestyle=":", label="Perfect")
            for model in combined["model"].unique():
                part = combined.loc[combined["model"].eq(model)]
                plt.plot(part["mean_predicted"], part["observed_rate"], marker="o", label=model)
            plt.xlabel("Mean predicted probability")
            plt.ylabel("Observed event rate")
            plt.legend(ncol=2)
            plt.tight_layout()
            plt.savefig(figures / f"{dataset}_reliability_diagram.pdf")
            plt.savefig(figures / f"{dataset}_reliability_diagram.png", dpi=400)
            plt.close()
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(source / "reliability_diagram.csv", index=False)


def run_dabra_interventions(frame, indices, out, workers):
    train_idx, val_idx, test_idx = indices
    processor = DatasetPreprocessor("dabra").fit(frame.iloc[train_idx])
    x_train, _, _ = processor.transform(frame.iloc[train_idx])
    x_test, _, _ = processor.transform(frame.iloc[test_idx])
    y_train = frame.iloc[train_idx]["target"].to_numpy()
    model = LogisticRegression(
        solver="saga",
        max_iter=500,
        class_weight="balanced",
        n_jobs=workers,
        random_state=11
    )
    model.fit(x_train, y_train)
    baseline = model.predict_proba(x_test)[:, 1].mean()
    rows = []
    for column in ["weather_conditions", "road_type", "traffic_condition"]:
        for value in sorted(frame.iloc[train_idx][column].astype(str).unique()):
            changed = frame.iloc[test_idx].copy()
            changed[column] = value
            x_changed, _, _ = processor.transform(changed)
            risk = model.predict_proba(x_changed)[:, 1].mean()
            rows.append({
                "variable": column,
                "intervention_value": value,
                "baseline_risk": baseline,
                "intervention_risk": risk,
                "risk_difference": risk - baseline
            })
    result = pd.DataFrame(rows)
    result.to_csv(out / "tables" / "dabra_structural_interventions.csv", index=False)
    x = np.arange(len(result))
    plt.figure(figsize=(7.16, 3.8))
    plt.bar(x, result["risk_difference"])
    plt.axhline(0, linestyle=":")
    plt.xticks(x, result["variable"] + "=" + result["intervention_value"], rotation=35, ha="right")
    plt.ylabel("Estimated change in unsafe-risk probability")
    plt.xlabel("Controlled intervention")
    plt.tight_layout()
    plt.savefig(out / "figures" / "dabra_structural_interventions.pdf")
    plt.savefig(out / "figures" / "dabra_structural_interventions.png", dpi=400)
    plt.close()


def run_veremi_block_sensitivity(frame, indices, out, workers):
    train_idx, val_idx, test_idx = indices
    processor = DatasetPreprocessor("veremi").fit(frame.iloc[train_idx])
    x_train, dims, names = processor.transform(frame.iloc[train_idx])
    x_test, _, _ = processor.transform(frame.iloc[test_idx])
    y_train = frame.iloc[train_idx]["target"].to_numpy()
    model = LogisticRegression(
        solver="saga",
        max_iter=500,
        class_weight="balanced",
        n_jobs=workers,
        random_state=11
    )
    model.fit(x_train, y_train)
    baseline = model.predict_proba(x_test)[:, 1].mean()
    rows = []
    offset = 0
    for name, dim in zip(names, dims):
        changed = x_test.copy()
        changed[:, offset:offset + dim] = 0
        risk = model.predict_proba(changed)[:, 1].mean()
        rows.append({
            "feature_block": name,
            "baseline_risk": baseline,
            "mean_replacement_risk": risk,
            "risk_difference": risk - baseline
        })
        offset += dim
    result = pd.DataFrame(rows)
    result.to_csv(out / "tables" / "veremi_feature_block_sensitivity.csv", index=False)
    x = np.arange(len(result))
    plt.figure(figsize=(5.5, 3.8))
    plt.bar(x, result["risk_difference"])
    plt.axhline(0, linestyle=":")
    plt.xticks(x, result["feature_block"], rotation=30, ha="right")
    plt.ylabel("Change in predicted fault risk")
    plt.xlabel("Feature block replaced by training mean")
    plt.tight_layout()
    plt.savefig(out / "figures" / "veremi_feature_block_sensitivity.pdf")
    plt.savefig(out / "figures" / "veremi_feature_block_sensitivity.png", dpi=400)
    plt.close()

def create_figures(metrics, out):
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    source = out / "figure_sources"
    source.mkdir(parents=True, exist_ok=True)
    test = metrics.loc[
        metrics["partition"].eq("test")
        & metrics["calibration"].eq("platt")
        & metrics["model"].isin(["Prevalence", "LR", "XGB", "ERM", "IRM", "DANN", "GroupDRO", "CORAL", "MMD", "CIDT"])
    ].copy()
    iid = test.loc[test["split"].eq("iid")]
    if not iid.empty:
        summary = iid.groupby(["dataset", "model"], as_index=False).agg(
            train_placeholder=("auroc", "mean"),
            test_auroc=("auroc", "mean"),
            test_auroc_std=("auroc", "std"),
            auprc_gain=("auprc_gain", "mean")
        )
        train = metrics.loc[
            metrics["partition"].eq("train")
            & metrics["calibration"].eq("platt")
            & metrics["split"].eq("iid")
        ].groupby(["dataset", "model"], as_index=False)["auroc"].mean().rename(columns={"auroc": "train_auroc"})
        summary = summary.drop(columns=["train_placeholder"]).merge(train, on=["dataset", "model"], how="left")
        summary.to_csv(source / "train_test_auroc.csv", index=False)
        for dataset in summary["dataset"].unique():
            part = summary.loc[summary["dataset"].eq(dataset)]
            x = np.arange(len(part))
            plt.figure(figsize=(7.16, 3.8))
            plt.plot(x, part["train_auroc"], marker="o", label="Train")
            plt.plot(x, part["test_auroc"], marker="s", linestyle="--", label="IID test")
            plt.axhline(0.5, linestyle=":")
            plt.xticks(x, part["model"], rotation=35, ha="right")
            plt.ylabel("AUROC")
            plt.xlabel("Model")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures / f"{dataset}_train_test_auroc.pdf")
            plt.savefig(figures / f"{dataset}_train_test_auroc.png", dpi=400)
            plt.close()
            plt.figure(figsize=(7.16, 3.8))
            plt.bar(x, part["auprc_gain"])
            plt.axhline(0, linestyle=":")
            plt.xticks(x, part["model"], rotation=35, ha="right")
            plt.ylabel("AUPRC minus prevalence")
            plt.xlabel("Model")
            plt.tight_layout()
            plt.savefig(figures / f"{dataset}_auprc_gain.pdf")
            plt.savefig(figures / f"{dataset}_auprc_gain.png", dpi=400)
            plt.close()
    ablation = metrics.loc[
        metrics["partition"].eq("test")
        & metrics["calibration"].eq("platt")
        & metrics["model"].str.startswith("CIDT")
    ].groupby(["dataset", "split", "model"], as_index=False).agg(
        auprc=("auprc", "mean"),
        ece=("ece", "mean"),
        fsr=("fsr", "mean")
    )
    ablation.to_csv(source / "ablation.csv", index=False)
    for dataset in ablation["dataset"].unique():
        part = ablation.loc[ablation["dataset"].eq(dataset)]
        if part.empty:
            continue
        x = np.arange(len(part))
        plt.figure(figsize=(7.16, 4.0))
        plt.bar(x, part["auprc"])
        plt.xticks(x, part["model"], rotation=35, ha="right")
        plt.ylabel("Test AUPRC")
        plt.xlabel("Ablation")
        plt.tight_layout()
        plt.savefig(figures / f"{dataset}_ablation_auprc.pdf")
        plt.savefig(figures / f"{dataset}_ablation_auprc.png", dpi=400)
        plt.close()
    sensitivity = metrics.loc[
        metrics["partition"].eq("test")
        & metrics["calibration"].eq("platt")
        & metrics["model"].str.startswith("IRM_lambda_")
    ].copy()
    sensitivity["lambda"] = sensitivity["model"].str.replace("IRM_lambda_", "", regex=False).astype(float)
    sensitivity = sensitivity.groupby(["dataset", "lambda"], as_index=False).agg(
        auprc=("auprc", "mean"),
        ece=("ece", "mean")
    )
    sensitivity.to_csv(source / "irm_sensitivity.csv", index=False)
    for dataset in sensitivity["dataset"].unique():
        part = sensitivity.loc[sensitivity["dataset"].eq(dataset)].sort_values("lambda")
        plt.figure(figsize=(5.5, 3.8))
        plt.semilogx(np.maximum(part["lambda"], 1e-3), part["auprc"], marker="o")
        plt.xlabel("IRM penalty coefficient")
        plt.ylabel("Test AUPRC")
        plt.tight_layout()
        plt.savefig(figures / f"{dataset}_irm_sensitivity.pdf")
        plt.savefig(figures / f"{dataset}_irm_sensitivity.png", dpi=400)
        plt.close()
    shifts = test.groupby(["dataset", "split", "model"], as_index=False).agg(
        auprc=("auprc", "mean"),
        ece=("ece", "mean"),
        fsr=("fsr", "mean")
    )
    shifts.to_csv(source / "shift_robustness.csv", index=False)
    for dataset in shifts["dataset"].unique():
        part = shifts.loc[shifts["dataset"].eq(dataset) & shifts["model"].isin(["XGB", "ERM", "IRM", "GroupDRO", "CIDT"])]
        if part.empty:
            continue
        pivot = part.pivot(index="split", columns="model", values="auprc")
        plt.figure(figsize=(7.16, 3.8))
        for model in pivot.columns:
            plt.plot(np.arange(len(pivot)), pivot[model], marker="o", label=model)
        plt.xticks(np.arange(len(pivot)), pivot.index, rotation=30, ha="right")
        plt.ylabel("Test AUPRC")
        plt.xlabel("Evaluation split")
        plt.legend(ncol=3)
        plt.tight_layout()
        plt.savefig(figures / f"{dataset}_shift_robustness.pdf")
        plt.savefig(figures / f"{dataset}_shift_robustness.png", dpi=400)
        plt.close()
    deployment = metrics.loc[
        metrics["partition"].eq("test")
        & metrics["model"].isin(["ERM", "IRM", "CIDT", "CIDT_INT8"])
    ].groupby(["dataset", "model"], as_index=False).agg(
        latency_median_ms=("latency_median_ms", "mean"),
        fsr=("fsr", "mean"),
        model_size_mb=("model_size_mb", "mean")
    )
    deployment.to_csv(source / "deployment.csv", index=False)
    for dataset in deployment["dataset"].unique():
        part = deployment.loc[deployment["dataset"].eq(dataset)]
        plt.figure(figsize=(5.5, 3.8))
        sizes = 40 + 20 * np.sqrt(np.maximum(part["model_size_mb"], 0))
        plt.scatter(part["latency_median_ms"], part["fsr"], s=sizes)
        for _, row in part.iterrows():
            plt.text(row["latency_median_ms"], row["fsr"], row["model"])
        plt.xlabel("Median latency per sample (ms)")
        plt.ylabel("False-safe rate")
        plt.tight_layout()
        plt.savefig(figures / f"{dataset}_deployment.pdf")
        plt.savefig(figures / f"{dataset}_deployment.png", dpi=400)
        plt.close()


def create_tables(metrics, out):
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    test = metrics.loc[metrics["partition"].eq("test") & metrics["calibration"].eq("platt")].copy()
    main = test.groupby(["dataset", "split", "model"], as_index=False).agg(
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        auprc_mean=("auprc", "mean"),
        auprc_std=("auprc", "std"),
        brier_mean=("brier", "mean"),
        ece_mean=("ece", "mean"),
        fsr_mean=("fsr", "mean"),
        mcc_mean=("mcc", "mean"),
        latency_ms=("latency_median_ms", "mean")
    )
    main.to_csv(tables / "main_results.csv", index=False)
    ablation = main.loc[main["model"].str.startswith("CIDT")]
    ablation.to_csv(tables / "ablation_results.csv", index=False)
    ood = main.loc[~main["split"].eq("iid")]
    ood.to_csv(tables / "ood_results.csv", index=False)
    training = metrics.loc[metrics["partition"].eq("train")].groupby(
        ["dataset", "split", "model"], as_index=False
    ).agg(
        train_auroc=("auroc", "mean"),
        train_auprc=("auprc", "mean"),
        training_seconds=("training_seconds", "mean")
    )
    training.to_csv(tables / "training_results.csv", index=False)


def environment_config(profile, fraction, device, workers):
    if profile == "quick":
        seeds = [11]
        epochs = 5
        patience = 2
        batch = 4096
        bootstrap = 100
        limit = 400000
    else:
        seeds = [11, 23, 37, 41, 53]
        epochs = 15
        patience = 5
        batch = 4096
        bootstrap = 500
        limit = 0
    return RunConfig(
        profile=profile,
        fraction=fraction,
        seeds=seeds,
        epochs=epochs,
        patience=patience,
        batch_size=batch,
        bootstrap=bootstrap,
        veremi_limit=limit,
        workers=workers,
        device=device
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dabra", default=r"D:\other\Causal-DT\Dataset\DABRA.csv")
    parser.add_argument("--veremi", default=r"D:\other\Causal-DT\Dataset\VeReMi.csv")
    parser.add_argument("--out", default=r"D:\other\Causal-DT\RevisionResults")
    parser.add_argument("--profile", choices=["quick", "paper"], default="quick")
    parser.add_argument("--fraction", type=float, default=0.50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--dataset", choices=["both", "dabra", "veremi"], default="both")
    args = parser.parse_args()
    out = Path(args.out)
    for directory in ["data", "models", "predictions", "history", "figures", "figure_sources", "tables", "preprocessors"]:
        (out / directory).mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    config = environment_config(args.profile, args.fraction, device, args.workers)
    save_json(out / "run_config.json", asdict(config))
    all_metrics = []
    if args.dataset in ["both", "dabra"]:
        dabra = load_dabra(args.dabra, config.fraction, config.seeds[0], out)
        save_json(out / "data" / "dabra_summary.json", {
            "rows": len(dabra),
            "target_counts": dabra["target"].value_counts().sort_index().to_dict(),
            "drivers": dabra["driver_id"].value_counts().sort_index().to_dict(),
            "environments": int(dabra["environment"].nunique())
        })
        splits = dabra_splits(dabra, config.seeds[0])
        run_dabra_interventions(dabra, splits["iid"], out, config.workers)
        metrics = run_dataset("dabra", dabra, splits, config, out)
        metrics.to_csv(out / "dabra_metrics.csv", index=False)
        all_metrics.append(metrics)
        del dabra, metrics
        gc.collect()
    if args.dataset in ["both", "veremi"]:
        veremi = load_veremi(args.veremi, config.fraction, config.seeds[0], out, config.veremi_limit)
        splits, mapping = veremi_splits(veremi, config.seeds[0])
        save_json(out / "data" / "veremi_split_mapping.json", mapping)
        save_json(out / "data" / "veremi_summary.json", {
            "rows": len(veremi),
            "target_counts": veremi["target"].value_counts().sort_index().to_dict(),
            "class_counts": veremi["class"].value_counts().sort_index().to_dict(),
            "senders": int(veremi["sender"].nunique()),
            "senderpseudos": int(veremi["senderpseudo"].nunique())
        })
        run_veremi_block_sensitivity(veremi, splits["iid"], out, config.workers)
        metrics = run_dataset("veremi", veremi, splits, config, out)
        metrics.to_csv(out / "veremi_metrics.csv", index=False)
        all_metrics.append(metrics)
        del veremi, metrics
        gc.collect()
    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        combined.to_csv(out / "all_metrics.csv", index=False)
        create_tables(combined, out)
        create_figures(combined, out)
        create_calibration_figures(out)
    print(f"Completed. Results saved to {out}")


if __name__ == "__main__":
    main()
