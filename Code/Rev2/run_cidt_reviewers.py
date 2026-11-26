import argparse
import sys
import datetime
import contextlib
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
import matplotlib
matplotlib.use("Agg")
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



@dataclass
class RunConfig:
    profile: str
    fraction: float
    seeds: list
    epochs: int
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

def calibration_fit(y, p):
    model = LogisticRegression(solver="lbfgs")
    model.fit(logit_clip(p).reshape(-1, 1), np.asarray(y, dtype=int))
    return model

def calibration_apply(model, p):
    return model.predict_proba(logit_clip(p).reshape(-1, 1))[:, 1]

def compute_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    p = np.clip(p, 1e-7, 1 - 1e-7)
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
    groups = [representation[env == v] for v in torch.unique(env)]
    groups = [g for g in groups if len(g) >= 4]
    if len(groups) < 2:
        return representation.new_zeros(())
    means = torch.stack([g.mean(0) for g in groups])
    covs = torch.stack([covariance(g) for g in groups])
    return 2 * (means.var(0, unbiased=True).mean() + covs.var(0, unbiased=True).mean())


def mmd_penalty(representation, env):
    means = []
    for value in torch.unique(env):
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
    if name in {"CIDT", "CIDT_no_IRM", "CIDT_no_CORAL", "CIDT_no_group_variance", "CIDT_no_auxiliary", "CIDT_group_risk_only"}:
        risk, irm = irm_penalty(logits, y, env)
        coral = coral_penalty(representation, env)
        variance = group_risk_variance(logits, y, env)
        weight = 1.0 if epoch < max(2, total_epochs // 4) else 10.0
        extra["irm"] = irm
        extra["coral"] = coral
        extra["variance"] = variance
        return (risk + (0 if name in {"CIDT_no_IRM", "CIDT_no_auxiliary", "CIDT_group_risk_only"} else weight) * irm
                + (0 if name in {"CIDT_no_CORAL", "CIDT_no_auxiliary", "CIDT_group_risk_only"} else 0.1) * coral
                + (0 if name in {"CIDT_no_group_variance", "CIDT_no_auxiliary", "CIDT_group_risk_only"} else 0.1) * variance), extra
    return base, extra

def predict_neural(model, x, device, batch=8192, return_representation=False):
    model.eval()
    probabilities = []
    representations = []
    tensor = torch.from_numpy(x.astype(np.float32, copy=False))
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

def compact_metrics(y, p):
    # A fixed 0.5 operating point for epoch monitoring; final calibrated
    # metrics and validation-selected operating points are reported separately.
    m = compute_metrics(y, p, 0.5)
    return {k: m[k] for k in ['nll','auroc','auprc','accuracy','balanced_accuracy','f1','mcc','brier','ece','fsr','fur']}


def print_metrics(prefix, m):
    fields = ['n','prevalence','nll','auroc','auprc','accuracy','balanced_accuracy','f1','mcc','brier','ece','fsr','fur','threshold']
    print(prefix + ' | ' + ' | '.join(f'{k}={m[k]:.6f}' if k != 'n' else f'n={m[k]}' for k in fields if k in m), flush=True)


def save_partition_metrics(root, stem, rows):
    atomic_csv(pd.DataFrame(rows), root/'runs'/stem/'partition_metrics.csv')
    for m in rows:
        print_metrics(f"FINAL | {stem} | {m['partition'].upper()} | {m['calibration']}", m)


def train_neural(name, x_train, y_train, env_train, x_val, y_val, branch_dims, branch_names, seed, config, disabled=None, irm_lambda=None, history_path=None, run_id=None):
    seed_everything(seed)
    device = torch.device(config.device)
    unique_domains = np.unique(env_train)
    domain_map = {value: index for index, value in enumerate(unique_domains)}
    mapped_env = np.array([domain_map[v] for v in env_train], dtype=np.int64)
    model = MultiBranchNet(branch_dims, branch_names, domains=max(1, len(unique_domains)), disabled=disabled).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = TensorDataset(torch.from_numpy(x_train.astype(np.float32, copy=False)), torch.from_numpy(y_train.astype(np.float32)), torch.from_numpy(mapped_env))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=False)
    history = []
    start = time.perf_counter()
    tag = run_id or name
    print(f'TRAINING | {tag} | seed={seed} | epochs={config.epochs} | early_stopping=OFF | checkpoint=last_epoch', flush=True)
    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        model.train()
        weighted_loss = 0.0
        for xb, yb, eb in loader:
            xb, yb, eb = xb.to(device), yb.to(device), eb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, representation = model(xb)
            loss, extras = neural_objective(name, logits, representation, yb, eb, model.domain_head, epoch, config.epochs)
            if irm_lambda is not None:
                risk, penalty = irm_penalty(logits, yb, eb)
                loss = risk + float(irm_lambda) * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss in {tag}, epoch {epoch+1}; no valid result can be reported')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            weighted_loss += float(loss.detach().cpu()) * len(yb)
        # Full training and validation predictions in evaluation mode.
        # Test predictions are intentionally not computed inside training.
        tr = compact_metrics(y_train, predict_neural(model, x_train, device))
        va = compact_metrics(y_val, predict_neural(model, x_val, device))
        mean_loss = weighted_loss / len(y_train)
        row = {'epoch': epoch+1, 'loss': mean_loss, 'train_objective_loss': mean_loss,
               'elapsed_epoch_seconds': time.perf_counter()-epoch_start,
               **{'train_'+k:v for k,v in tr.items()}, **{'validation_'+k:v for k,v in va.items()}}
        history.append(row)
        print(f'EPOCH | {tag} | {epoch+1}/{config.epochs} | objective_loss={mean_loss:.6f} | seconds={row["elapsed_epoch_seconds"]:.2f}', flush=True)
        print_metrics('  TRAIN (threshold=0.5)', tr)
        print_metrics('  VALIDATION (threshold=0.5)', va)
        if history_path is not None:
            atomic_csv(pd.DataFrame(history), history_path)
    # Use the final epoch, regardless of the validation trajectory.
    return model, pd.DataFrame(history), time.perf_counter()-start


def fit_sklearn(name, x_train, y_train, seed, workers):
    if name == "Constant05":
        model = ConstantModel(0.5)
    elif name == "Prevalence":
        model = PrevalenceModel()
    elif name == "LR":
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=120,
            random_state=seed
        )
    elif name == "XGB":
        model = XGBClassifier(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.07,
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
            early_stopping=False,
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

def select_branch_columns(x, branch_dims, branch_names, keep_name):
    offset = 0
    for dim, name in zip(branch_dims, branch_names):
        if name == keep_name:
            return x[:, offset:offset + dim]
        offset += dim
    raise ValueError(f"Branch {keep_name} not found in {branch_names}")

# ------------------------ Reviewer revision orchestration ------------------------
from scipy.stats import t as student_t
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import platform, shutil, copy, traceback, hashlib

SEEDS = [5, 11, 23, 37, 45]
MAIN_MODELS = ['Constant05','Prevalence','LR','LR_ShuffledY','XGB','ERM','IRM','DANN','GroupDRO','CORAL','MMD','CIDT']
ABLATIONS = ['CIDT_no_IRM','CIDT_no_CORAL','CIDT_no_group_variance','CIDT_no_auxiliary']
CORE = {'dabra':'environment_full','veremi':'unseen_attack'}
GROUP_COLUMNS = {'dabra':{'environment_coarse':['weather_conditions'], 'environment_medium':['weather_conditions','road_type'], 'environment_full':['weather_conditions','road_type','traffic_condition'], 'driver_heldout':['driver_id']}, 'veremi':{'sender_heldout':['sender']}}
REVIEW = {
 'R1_C01_OOD_consistency':['Table_S11_raw_test_metrics','Table_04_IID_OOD','Figure_03a_DABRA_OOD','Figure_03b_VeReMi_OOD','Table_S01_split_manifest','Table_S02_legacy_OOD_values'],
 'R1_C02_objective_ablation':['Table_05_loss_ablation','Figure_04a_DABRA_ablation','Figure_04b_VeReMi_ablation','Table_S03_paired_ablation_deltas'],
 'R1_C03_VeReMi_sensitivity':['Figure_05b_feature_sensitivity','Table_S04_feature_sensitivity'],
 'R1_C04_seeds_uncertainty':['Table_S12_all_partition_metrics','Table_S13_all_partition_per_seed','Table_03_dataset_protocol','Table_S05_per_seed_metrics','Table_S06_uncertainty','Table_S07_group_bootstrap','Table_04a_IID_predictive','Table_04b_IID_calibration_safety','Figure_S02_reliability'],
 'R1_C05_FP32_INT8':['Table_06_deployment','Figure_06_deployment','Figure_S01_FSR_FUR','Table_S08_deployment_paired'],
 'R1_C06_integrated_SEFN':['Table_07_scenario_component_ablation','Table_08_scenario_interventions','Figure_07_scenarios','Table_S09_standalone_module_results','Table_S10_synthetic_SEFN'],
 'R2_C01_OOD_consistency':['Table_04_IID_OOD','Figure_03a_DABRA_OOD','Figure_03b_VeReMi_OOD','Table_S01_split_manifest'],
 'R2_C02_component_ablation':['Table_07_scenario_component_ablation','Table_S09_standalone_module_results','Figure_07_scenarios'],
 'R2_C03_driving_decisions':['Table_07_scenario_component_ablation','Table_08_scenario_interventions','Figure_07_scenarios'],
 'R2_C04_causal_validation':['Table_08_scenario_interventions','Figure_05a_causal_graph','Figure_08_interventions','Table_S04_feature_sensitivity'],
 'R3_C01_quantization':['Table_06_deployment','Table_S08_deployment_paired','Figure_06_deployment'],
}

def choose_threshold(y,p,max_fsr=.05):
    # O(n log n), evaluates every distinct score + always-safe / always-unsafe.
    y=np.asarray(y,int); p=np.asarray(p,float)
    order=np.argsort(-p,kind='stable'); ps=p[order]; ys=y[order]
    ends=np.r_[np.where(ps[1:]!=ps[:-1])[0],len(ps)-1]
    tp=np.r_[0,np.cumsum(ys)[ends]]; fp=np.r_[0,np.cumsum(1-ys)[ends]]
    th=np.r_[np.nextafter(ps[0],np.inf),ps[ends]]
    fsr=1-tp/y.sum(); fur=fp/(len(y)-y.sum()); bacc=1-(fsr+fur)/2
    idx=np.lexsort((th,-fur,bacc,(fsr<=max_fsr+1e-12).astype(int)))[-1]
    return float(th[idx])


def json_write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp'); temp.write_text(json.dumps(obj,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8'); os.replace(temp,path)


def atomic_csv(df,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    df.to_csv(temp,index=False,compression='gzip' if path.suffix=='.gz' else None)
    os.replace(temp,path)


def latex_escape(x):
    m={'\\':r'\textbackslash{}','_':r'\_','%':r'\%','&':r'\&','#':r'\#','$':r'\$','{':r'\{','}':r'\}','^':r'\textasciicircum{}','~':r'\textasciitilde{}'}
    return ''.join(m.get(c,c) for c in str(x))


def publish_table(root,name,df,caption,columns=None,latex=True):
    dest=root/'paper'/'tables'; dest.mkdir(parents=True,exist_ok=True)
    atomic_csv(df,dest/(name+'.csv'))
    (dest/(name+'_caption.txt')).write_text(caption,encoding='utf-8')
    if not latex: return
    view=df if columns is None else df[[c for c in columns if c in df.columns]]
    v=view.copy()
    for c in v.select_dtypes(include='number'):
        v[c]=v[c].map(lambda z: '' if pd.isna(z) else (str(int(z)) if isinstance(z,(int,np.integer)) else f'{z:.4f}'))
    v=v.fillna('').astype(str)
    for c in v: v[c]=v[c].map(latex_escape)
    v.columns=[latex_escape(c) for c in v.columns]
    # Plain tabular, suitable for wrapping into the manuscript's table* environment.
    # Long supplementary tables are CSV-only, to avoid an oversized LaTeX float.
    if len(v)<=150 and len(v.columns)<=14:
        body=v.to_latex(index=False,escape=False)
        (dest/(name+'.tex')).write_text('% '+caption+'\n'+body,encoding='utf-8')
    (dest/(name+'_caption.txt')).write_text(caption,encoding='utf-8')


def save_plot(root,name,fig):
    path=root/'paper'/'figures'; path.mkdir(parents=True,exist_ok=True)
    fig.tight_layout(); fig.savefig(path/(name+'.png'),dpi=300,bbox_inches='tight'); fig.savefig(path/(name+'.pdf'),bbox_inches='tight'); plt.close(fig)


def load_data(dataset,path,limit,seed):
    if dataset=='dabra':
        f=pd.read_csv(path); f.columns=[snake(c) for c in f.columns]
        req=['anomalous_event','route_anomaly','geofencing_violation','stop_events']
        if f[req].isna().any().any(): raise ValueError('Missing target source values; resolve before running')
        f['target']=(f[req[:3]].eq(1).any(axis=1)|f.stop_events.ge(3)).astype('int8')
        f['timestamp']=pd.to_datetime(f.timestamp,errors='raise'); f['time_value']=f.timestamp.astype('int64')
    else:
        header=pd.read_csv(path,nrows=0); cmap={snake(c):c for c in header.columns}
        want=['sendtime','sender','class']+sum(DatasetPreprocessor('veremi').branches.values(),[])
        missing=set(want)-set(cmap)
        if missing: raise ValueError('Missing VeReMi columns: '+str(sorted(missing)))
        types={cmap[c]:'float32' for c in want if c not in ('sender','class','sendtime')}
        f=pd.read_csv(path,usecols=[cmap[c] for c in want],dtype=types); f.columns=[snake(c) for c in f.columns]
        f['class']=pd.to_numeric(f['class'],errors='raise'); f['target']=f['class'].ne(0).astype('int8')
        f['time_value']=pd.to_numeric(f.sendtime,errors='raise')
    f['source_row']=np.arange(len(f),dtype=np.int64)
    source={'dataset':dataset,'source_rows':len(f),'source_positive':int(f.target.sum()),'source_prevalence':float(f.target.mean()),'source_columns':len(pd.read_csv(path,nrows=0).columns)}
    if limit and len(f)>limit:
        # Stratify by attack subtype where available. Sampling happens before split creation.
        strat=f['class'] if dataset=='veremi' else f.target
        idx,_=train_test_split(np.arange(len(f)),train_size=limit,stratify=strat,random_state=seed)
        f=f.iloc[idx].sort_values('source_row').reset_index(drop=True)
    needed=sum(DatasetPreprocessor(dataset).branches.values(),[])
    if f[needed+['time_value','target']].isna().any().any(): raise ValueError(f'{dataset}: missing required feature/time values; no silent imputation')
    for cols in GROUP_COLUMNS[dataset].values():
        if f[cols].isna().any().any(): raise ValueError(f'{dataset}: missing group identifiers')
    if dataset=='dabra':
        f['environment_id']=pd.factorize(f[['weather_conditions','road_type','traffic_condition']].astype(str).agg('|'.join,axis=1))[0]
    else:
        # Training-only time-bin edges are computed separately for each split, below.
        f['environment_id']=0
    source.update(analysis_rows=len(f),analysis_positive=int(f.target.sum()),analysis_prevalence=float(f.target.mean()))
    return f,source


def select_rows(pool,y,n,seed):
    pool=np.asarray(pool,dtype=np.int64); n=min(int(n),len(pool))
    if n==len(pool): return pool.copy()
    if n<2 or np.unique(y[pool]).size!=2: raise ValueError('Partition cannot support binary stratification')
    try: a,_=train_test_split(pool,train_size=n,stratify=y[pool],random_state=seed); return a
    except ValueError: raise ValueError('Insufficient class support for requested split')


def group_partition(y,g,ntr,nv,nt,seed):
    # Group-disjoint test; closest feasible whole-group pool, then fixed-size sampling.
    codes,labels=pd.factorize(g,sort=True); counts=np.bincount(codes); rng=np.random.default_rng(seed)
    best=None
    for _ in range(150):
        perm=rng.permutation(len(labels)); cum=np.cumsum(counts[perm]); k=int(np.searchsorted(cum,nt))+1
        if k>=len(labels): continue
        chosen=perm[:k]; test_mask=np.isin(codes,chosen)
        te=np.flatnonzero(test_mask); rest=np.flatnonzero(~test_mask)
        if len(rest)<nv+2 or np.unique(y[te]).size<2 or np.unique(y[rest]).size<2: continue
        score=(max(0,ntr+nv-len(rest)),abs(len(te)-nt))
        if best is None or score<best[0]: best=(score,te,rest,chosen)
        if score==(0,0): break
    if best is None: raise ValueError('No feasible group-disjoint split with both classes')
    _,te,rest,chosen=best
    tr=select_rows(rest,y,min(ntr,len(rest)-nv),seed+1); va=select_rows(np.setdiff1d(rest,tr),y,nv,seed+2); te=select_rows(te,y,nt,seed+3)
    return (tr,va,te),{'heldout_groups':[str(labels[i]) for i in chosen]}


def split_data(dataset,f,args):
    y=f.target.to_numpy(); n=len(f); ntr=int(n*args.train_fraction); nv=int(n*.10); nt=int(n*.10); seed=args.split_seed
    allidx=np.arange(n); tr=select_rows(allidx,y,ntr,seed); remaining=np.setdiff1d(allidx,tr); va=select_rows(remaining,y,nv,seed+1); te=select_rows(np.setdiff1d(remaining,va),y,nt,seed+2)
    out={'iid':((tr,va,te),{'protocol':'random row-level stratified IID'})}
    # Cut on unique timestamps; never assign equal timestamps to different sets.
    times=f.time_value.to_numpy(); order=np.argsort(times,kind='stable'); sorted_t=times[order]
    cuts=[n if k>=n else np.searchsorted(sorted_t,sorted_t[k],side='left') for k in [ntr,ntr+nv,ntr+nv+nt]]
    a,b,c=cuts
    out['temporal']=((order[:a],order[a:b],order[b:c]),{'protocol':'strict chronological; tied timestamps remain together'})
    for name,cols in GROUP_COLUMNS[dataset].items():
        g=f[cols[0]].astype(str).to_numpy() if len(cols)==1 else f[cols].astype(str).agg('|'.join,axis=1).to_numpy()
        try: out[name]=group_partition(y,g,ntr,nv,nt,seed)
        except ValueError as e: out[name]=(None,{'error':str(e)})
    if dataset=='veremi':
        # Pick held-out attack classes with enough examples; disjoint normal rows.
        classes=f['class'].to_numpy(); attack,counts=np.unique(classes[classes!=0],return_counts=True)
        rng=np.random.default_rng(seed); best=None; normal=np.flatnonzero(classes==0); rng.shuffle(normal)
        if len(attack)>=2:
            for _ in range(200):
                perm=rng.permutation(len(attack)); k=max(1,round(len(attack)*.2)); held=attack[perm[:k]]
                hp=np.flatnonzero(np.isin(classes,held)); ap=np.flatnonzero((classes!=0)&~np.isin(classes,held)); rate=y.mean()
                test_a=min(len(hp),max(1,round(nt*rate))); test_n=nt-test_a
                max_dev=min(ntr+nv,int(len(ap)/rate),int((len(normal)-test_n)/(1-rate)))
                if test_n<1 or max_dev<nv+2 or test_n>len(normal): continue
                score=(ntr+nv-max_dev,abs(test_a-round(nt*rate)))
                if best is None or score<best[0]: best=(score,held,hp,ap,test_a,test_n,max_dev)
                if score==(0,0): break
        if best:
            _,held,hp,ap,ta,tn,dev=best
            te=np.r_[rng.choice(hp,ta,replace=False),normal[:tn]]
            pool=np.r_[ap,normal[tn:]]; tr=select_rows(pool,y,min(ntr,dev-nv),seed); va=select_rows(np.setdiff1d(pool,tr),y,nv,seed+1)
            out['unseen_attack']=((tr,va,te),{'heldout_attack_classes':held.tolist(),'protocol':'test attack types absent from training and validation; normal rows disjoint'})
        else: out['unseen_attack']=(None,{'error':'No feasible unseen-attack split'})
    for name,(parts,meta) in list(out.items()):
        if parts is None: continue
        if any(len(p)<2 or np.unique(y[p]).size!=2 for p in parts): out[name]=(None,{**meta,'error':'One partition lacks both classes'}); continue
        if len(np.unique(np.concatenate(parts)))!=sum(map(len,parts)): raise AssertionError('Overlapping rows')
        if name in GROUP_COLUMNS[dataset]:
            cols=GROUP_COLUMNS[dataset][name]; groups=lambda idx:set(map(tuple,f.iloc[idx][cols].to_numpy()))
            assert not groups(parts[2]) & groups(np.r_[parts[0],parts[1]])
        if name=='unseen_attack': assert not (set(f.iloc[parts[2]]['class'])-{0}) & set(f.iloc[np.r_[parts[0],parts[1]]]['class'])
    return out


def group_bootstrap(y,p,groups,nboot,seed,cap=30000):
    # Cluster bootstrap conditional on a fitted model; no claim of independent seeds.
    if nboot<=0: return {}
    rng=np.random.default_rng(seed); y=np.asarray(y);p=np.asarray(p); groups=np.asarray(groups)
    if len(y)>cap:
        idx=rng.choice(len(y),cap,replace=False); y,p,groups=y[idx],p[idx],groups[idx]
    codes,labels=pd.factorize(groups); order=np.argsort(codes); blocks=np.split(order,np.cumsum(np.bincount(codes))[:-1]); results=[]
    if len(blocks)<2: return {'bootstrap_groups':len(blocks),'bootstrap_valid':0,'bootstrap_rows':len(y),'bootstrap_requested':nboot}
    for _ in range(nboot):
        ix=np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),len(blocks))]); yy=y[ix]; pp=p[ix]
        if np.unique(yy).size==2: results.append((roc_auc_score(yy,pp),average_precision_score(yy,pp),brier_score_loss(yy,pp)))
    row={'bootstrap_groups':len(blocks),'bootstrap_valid':len(results),'bootstrap_rows':len(y),'bootstrap_requested':nboot}
    if len(results)>=max(2,nboot//2):
        for i,key in enumerate(['auroc','auprc','brier']): row[key+'_boot_low'],row[key+'_boot_high']=np.quantile(np.asarray(results)[:,i],[.025,.975])
    return row


def seed_summary(df,keys,metrics):
    rows=[]
    for key,g in df.groupby(keys,dropna=False,sort=False):
        key=key if isinstance(key,tuple) else (key,); row=dict(zip(keys,key)); row['runs']=g.seed.nunique(); row['seeds']=','.join(map(str,sorted(g.seed.unique())))
        for m in metrics:
            vals=pd.to_numeric(g[m],errors='coerce').dropna() if m in g else pd.Series(dtype=float); k=len(vals); mean=vals.mean(); sd=vals.std(ddof=1)
            row[m+'_mean']=mean; row[m+'_sd']=sd
            width=student_t.ppf(.975,k-1)*sd/np.sqrt(k) if k>1 else np.nan
            row[m+'_t95_low']=mean-width; row[m+'_t95_high']=mean+width
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_row(y,p,threshold,**meta):
    return {**meta,**compute_metrics(y,p,threshold),'unsafe_prediction_rate':float(np.mean(p>=threshold))}


def fit_cal(y,p):
    return calibration_fit(y,p)


def pred_single(model,neural,x,device):
    return predict_neural(model,x,torch.device(device)) if neural else predict_sklearn(model,x)


def dump_preds(root,stem,source,y,p,pc,th,tcal):
    atomic_csv(pd.DataFrame({'source_row':source,'target':y,'raw_probability':p,'calibrated_probability':pc,'raw_threshold':th,'calibrated_threshold':tcal,'raw_prediction':p>=th,'calibrated_prediction':pc>=tcal}),root/'predictions'/(stem+'.csv.gz'))


def deployment(root,stem,model,xv,xt,yv,yt,cal,rawth,calth,meta,args,source):
    # Dynamic INT8 weights + dynamically quantized activations in Linear layers only.
    cpu=copy.deepcopy(model).cpu().eval(); ppv=predict_neural(cpu,xv,torch.device('cpu')); ppt=predict_neural(cpu,xt,torch.device('cpu'))
    try:
        engines=torch.backends.quantized.supported_engines
        if 'x86' in engines: torch.backends.quantized.engine='x86'
        elif 'fbgemm' in engines: torch.backends.quantized.engine='fbgemm'
        elif 'qnnpack' in engines: torch.backends.quantized.engine='qnnpack'
        quant=torch.ao.quantization.quantize_dynamic(copy.deepcopy(cpu),{nn.Linear},dtype=torch.qint8)
        qv=predict_neural(quant,xv,torch.device('cpu')); qt=predict_neural(quant,xt,torch.device('cpu'))
    except Exception as e: raise RuntimeError('INT8 quantization failed; no fabricated comparison') from e
    rows=[]; paths=root/'models'; paths.mkdir(exist_ok=True)
    for precision,m,pv,pt in [('FP32',cpu,ppv,ppt),('INT8',quant,qv,qt)]:
        path=paths/(stem+'_'+precision+'_cpu.pt'); torch.save(m.state_dict(),path); sz=path.stat().st_size/1048576
        med,p95=latency_neural(m,xt,torch.device('cpu'),repeats=args.latency_repeats)
        # Primary calibrated comparison shares the original FP32 calibrator and threshold.
        # An independent INT8 validation operating point is a secondary analysis only.
        pc=calibration_apply(cal,pt); altcal=fit_cal(yv,pv); altv=calibration_apply(altcal,pv); altp=calibration_apply(altcal,pt); altth=choose_threshold(yv,altv)
        for setting,p,th in [('shared_raw',pt,rawth),('shared_platt',pc,calth),('own_validation_platt',altp,altth)]:
            rows.append(metrics_row(yt,p,th,**meta,precision=precision,setting=setting,model_size_mib=sz,latency_median_ms=med,latency_p95_ms=p95,throughput_single_sample_per_s=1000/max(med,1e-9),cpu_threads=torch.get_num_threads(),engine=torch.backends.quantized.engine))
        dump_preds(root,stem+'_'+precision+'_deployment',source,yt,pt,pc,rawth,calth)
        joblib.dump(altcal,paths/(stem+'_'+precision+'_own_calibrator.joblib'))
    return rows


def run_one(root,dataset,split,seed,label,model_name,xtr,xv,xt,ytr,yv,yt,env,dims,names,config,args,source,bootgroups,disabled=None,lam=None):
    stem=f'{dataset}__{split}__{label}__s{seed}'; dest=root/'runs'/stem; dest.mkdir(parents=True,exist_ok=True)
    marker=dest/'complete.json'
    if marker.exists() and not args.no_resume:
        print('RESUME completed',stem,flush=True)
        return json.loads(marker.read_text(encoding='utf-8'))
    print('RUN',stem,flush=True); seed_everything(seed); st=time.perf_counter(); train_y=ytr
    models_dir=root/'models'; models_dir.mkdir(exist_ok=True)
    branch={'LR_BehaviorOnly':'behavior','LR_EnvironmentOnly':'environment','LR_PositionOnly':'position','LR_SpeedOnly':'speed'}
    neural=model_name not in set(['Constant05','Prevalence','LR','LR_ShuffledY','XGB']+list(branch))
    if model_name in branch: xtr,xv,xt=[select_branch_columns(x,dims,names,branch[model_name]) for x in (xtr,xv,xt)]
    if model_name=='LR_ShuffledY': train_y=np.random.default_rng(seed).permutation(ytr)
    if neural:
        model,hist,_=train_neural(model_name,xtr,ytr,env,xv,yv,dims,names,seed,config,disabled=disabled,irm_lambda=lam,history_path=dest/'history.csv',run_id=stem)
        atomic_csv(hist,dest/'history.csv'); best_epoch=int(hist.loc[hist.validation_auprc.idxmax(),'epoch']); path=models_dir/(stem+'.pt'); torch.save(model.state_dict(),path); json_write(dest/'architecture.json',{'branch_dims':dims,'branch_names':names,'domains':len(np.unique(env)),'hidden':64,'dropout':.15,'disabled':disabled or []})
    else:
        model,_=fit_sklearn('LR' if model_name.startswith('LR') else model_name,xtr,train_y,seed,args.workers); best_epoch=0; path=models_dir/(stem+'.joblib'); joblib.dump(model,path)
    elapsed=time.perf_counter()-st
    epochs_completed=len(hist) if neural else 0
    pv=pred_single(model,neural,xv,config.device); pt=pred_single(model,neural,xt,config.device); ptr=pred_single(model,neural,xtr,config.device)
    cal=fit_cal(yv,pv); pcv=calibration_apply(cal,pv); pct=calibration_apply(cal,pt); rawth=choose_threshold(yv,pv); calth=choose_threshold(yv,pcv)
    joblib.dump(cal,models_dir/(stem+'_calibrator.joblib'))
    meta=dict(dataset=dataset,split=split,model=label,seed=seed)
    # Matched CPU forward timing for all models; excludes preprocessing/calibration.
    med,p95=latency_neural(model,xt,torch.device('cpu'),args.latency_repeats) if neural else latency_sklearn(model,xt,args.latency_repeats)
    rows=[]
    for part,y,p,pc in [('train',ytr,ptr,calibration_apply(cal,ptr)),('validation',yv,pv,pcv),('test',yt,pt,pct)]:
        for calibration,prob,th in [('raw',p,rawth),('platt',pc,calth)]:
            rows.append(metrics_row(y,prob,th,**meta,partition=part,calibration=calibration,training_seconds=elapsed,best_epoch=best_epoch,epochs_completed=epochs_completed,selected_epoch=epochs_completed,model_size_mib=path.stat().st_size/1048576,latency_median_ms=med,latency_p95_ms=p95))
    save_partition_metrics(root,stem,rows)
    dump_preds(root,stem,source,yt,pt,pct,rawth,calth)
    boot=[]; deploy=[]; sensitivity=[]; components=[]
    if label in ['CIDT','ERM','IRM'] and split in ['iid',CORE[dataset]]:
        boot=[{**meta,**group_bootstrap(yt,pct,bootgroups,args.bootstrap,seed)}]
    if label=='CIDT' and split in ['iid',CORE[dataset]]:
        deploy=deployment(root,stem,model,xv,xt,yv,yt,cal,rawth,calth,meta,args,source)
    if label=='CIDT':
        # Structural readout from the fitted latent representation. No fault/action
        # variable exists jointly with the outcome in these datasets.
        ptr,ztr=predict_neural(model,xtr,torch.device('cpu'),return_representation=True)
        _,zv=predict_neural(model,xv,torch.device('cpu'),return_representation=True)
        _,zt=predict_neural(model,xt,torch.device('cpu'),return_representation=True)
        sctm=LogisticRegression(max_iter=300,random_state=seed).fit(ztr,ytr)
        sv=sctm.predict_proba(zv)[:,1]; sp=sctm.predict_proba(zt)[:,1]; sc=fit_cal(yv,sv); scv=calibration_apply(sc,sv); scp=calibration_apply(sc,sp); sth=choose_threshold(yv,scv)
        # This is a reduced, observational SCTM readout, not an integrated model.
        for part, yy, raw in [('train',ytr,sctm.predict_proba(ztr)[:,1]),('validation',yv,sv),('test',yt,sp)]:
            for kind, prob, threshold in [('raw',raw,choose_threshold(yv,sv)),('platt',calibration_apply(sc,raw),sth)]:
                r=metrics_row(yy,prob,threshold,**{**meta,'model':'SCTM_latent_readout'},partition=part,calibration=kind,training_seconds=np.nan,best_epoch=best_epoch,epochs_completed=epochs_completed,selected_epoch=epochs_completed,model_size_mib=np.nan,latency_median_ms=np.nan,latency_p95_ms=np.nan)
                rows.append(r); print_metrics(f'FINAL | {stem} | SCTM | {part} | {kind}',r)
        joblib.dump({'readout':sctm,'calibrator':sc,'threshold':sth},models_dir/(stem+'_SCTM_readout.joblib'))
        if split==CORE[dataset]:
            del ztr,zv,zt,ptr
            # Controlled mean replacement of actual observed input features only.
            # Paired probability differences, not causal effects.
            offset=0; blocks=[]
            if dataset=='veremi':
                for b,d in zip(names,dims): blocks.append((b,list(range(offset,offset+d)))); offset+=d
            else:
                cols=DatasetPreprocessor(dataset).branches['behavior']
                for b in ['speed','acceleration','steering_angle','brake_usage','lane_deviation','behavioral_consistency_index','route_deviation_score']:
                    blocks.append((b,[cols.index(b)]))
            baseline=calibration_apply(cal,predict_neural(model,xt,torch.device('cpu')))
            for b,ix in blocks:
                changed=xt.copy(); changed[:,ix]=xtr[:,ix].mean(0); p2=calibration_apply(cal,predict_neural(model,changed,torch.device('cpu'))); delta=p2-baseline
                sensitivity.append({**meta,'feature':b,'baseline_risk':baseline.mean(),'perturbed_risk':p2.mean(),'mean_risk_change':delta.mean(),'mean_absolute_change':np.abs(delta).mean(),'auroc_change':roc_auc_score(yt,p2)-roc_auc_score(yt,baseline)})
    output={'metrics':rows,'bootstrap':boot,'deployment':deploy,'sensitivity':sensitivity,'components':components}
    atomic_csv(pd.DataFrame(rows),dest/'partition_metrics.csv')
    json_write(marker,output); return output


def run_real(root,dataset,path,config,args):
    f,stats=load_data(dataset,path,args.dabra_limit if dataset=='dabra' else args.veremi_limit,args.split_seed); n=len(f)
    print('DATASET | '+json.dumps(stats,default=str),flush=True)
    atomic_csv(pd.DataFrame([stats]),root/'raw'/f'{dataset}_dataset_statistics.csv')
    splits=split_data(dataset,f,args); manifest=[]; result={k:[] for k in ['metrics','bootstrap','deployment','sensitivity','components']}
    for split,(parts,info) in splits.items():
        if parts is None:
            manifest.append(dict(dataset=dataset,split=split,status='unavailable',details=json.dumps(info))); print('SKIP',dataset,split,info,flush=True); continue
        tr,va,te=parts; y=f.target.to_numpy(); source=f.source_row.to_numpy()
        splitdir=root/'splits'/dataset/split; splitdir.mkdir(parents=True,exist_ok=True)
        split_stats=[]; subtype_stats=[]
        for label,idx in zip(['train','validation','test'],parts):
            record=dict(dataset=dataset,split=split,status='complete',partition=label,n=len(idx),positive=int(y[idx].sum()),negative=int(len(idx)-y[idx].sum()),prevalence=float(y[idx].mean()),fraction=len(idx)/n,details=json.dumps(info))
            manifest.append(record); split_stats.append(record)
            print(f'DATA SPLIT | {dataset} | {split} | {label.upper()} | rows={len(idx):,} | negative={record["negative"]:,} | positive={record["positive"]:,} | positive_rate={record["prevalence"]:.6f} | fraction={record["fraction"]:.6f}',flush=True)
            if dataset=='veremi':
                for cl,count in f.iloc[idx]['class'].value_counts().sort_index().items(): subtype_stats.append(dict(partition=label,attack_class=cl,n=count,fraction=count/len(idx)))
            atomic_csv(pd.DataFrame({'source_row':source[idx],'target':y[idx]}),splitdir/(label+'.csv.gz'))
        print(f'DATA SPLIT | {dataset} | {split} | UNUSED | rows={n-sum(map(len,parts)):,}',flush=True)
        atomic_csv(pd.DataFrame(split_stats),splitdir/'dataset_statistics.csv')
        if subtype_stats:
            atomic_csv(pd.DataFrame(subtype_stats),splitdir/'attack_class_statistics.csv')
            print(pd.DataFrame(subtype_stats).to_string(index=False),flush=True)
        atomic_csv(pd.DataFrame(manifest),root/'raw'/f'{dataset}_split_statistics_partial.csv')
        proc=DatasetPreprocessor(dataset).fit(f.iloc[tr]); joblib.dump(proc,splitdir/'preprocessor.joblib')
        xtr,dims,names=proc.transform(f.iloc[tr]); xv,_,_=proc.transform(f.iloc[va]); xt,_,_=proc.transform(f.iloc[te]);
        if not all(np.isfinite(x).all() for x in (xtr,xv,xt)): raise ValueError('Non-finite transformed values')
        if dataset=='veremi':
            edges=np.unique(np.quantile(f.time_value.to_numpy()[tr],np.linspace(0,1,9)))[1:-1]; env=np.searchsorted(edges,f.time_value.to_numpy()[tr]); json_write(splitdir/'environment_bins.json',edges.tolist()); bootgroups=f.sender.to_numpy()[te]
        else: env=f.environment_id.to_numpy()[tr]; bootgroups=f.driver_id.to_numpy()[te]
        models=MAIN_MODELS+(['LR_BehaviorOnly','LR_EnvironmentOnly'] if dataset=='dabra' else ['LR_PositionOnly','LR_SpeedOnly']) if split=='iid' else ['Constant05','Prevalence','LR','XGB','ERM','IRM','GroupDRO','CIDT']
        if args.profile=='smoke': models=['Constant05','ERM','CIDT']
        specs=[(m,m,None,1.) for m in models]
        if split==CORE[dataset]:
            specs += [(m,m,None,1.) for m in ABLATIONS]
            specs += [('IRM_lambda_'+str(v),'IRM',v,1.) for v in [0,.1,1,10,100]]
            specs += [(f'CIDT_fraction_{v:g}','CIDT',None,v) for v in [.1,.25,.5,.75]]
        for seed in config.seeds:
            for label,m,lam,fraction in specs:
                if fraction!=1:
                    rng=np.random.default_rng(args.split_seed); keep=[]
                    for cl in [0,1]:
                        pool=np.flatnonzero(y[tr]==cl); perm=rng.permutation(pool); keep.extend(perm[:max(1,int(len(pool)*fraction))])
                    idx=np.array(keep); xx=xtr[idx]; yy=y[tr][idx]; ee=env[idx]
                else: xx=xtr; yy=y[tr]; ee=env
                print(f'MODEL DATA | {dataset} | {split} | {label} | seed={seed} | train={len(yy):,} (negative={int(len(yy)-yy.sum()):,}, positive={int(yy.sum()):,}) | validation={len(va):,} | test={len(te):,}',flush=True)
                o=run_one(root,dataset,split,seed,label,m,xx,xv,xt,yy,y[va],y[te],ee,dims,names,config,args,source[te],bootgroups,lam=lam)
                for k in result: result[k].extend(o[k])
                atomic_csv(pd.DataFrame(result['metrics']),root/'raw'/f'{dataset}_metrics_partial.csv')
        del xtr,xv,xt; gc.collect()
    return result,stats,manifest

SCENARIOS=['Emergency braking','Lane keeping','V2X message validation']

def simulate(seed,n):
    # Explicit synthetic benchmark; not a physical driving simulator.
    rng=np.random.default_rng(seed)
    scenario=rng.integers(0,3,n); context=rng.normal(0,1,n); hazard=rng.normal(0,1,n); env=rng.integers(0,4,n)
    fault=rng.binomial(1,sigmoid_np(-.6+.65*context+.25*env)); diagnostic=1.9*fault+rng.normal(0,.85,n)+.15*context
    action=rng.binomial(1,.5,n); u=rng.random(n)
    def oracle(a,f): return sigmoid_np(-.7+.95*hazard+.35*context+1.25*f + np.array([-1.65,-1.2,-1.0])[scenario]*a + .2*(scenario==2))
    p=oracle(action,fault); y=(u<p).astype(int)
    state=np.c_[hazard,context,np.eye(3)[scenario]].astype('float32'); diag=np.c_[diagnostic,context].astype('float32')
    return {'scenario':scenario,'environment':env,'state':state,'diag':diag,'fault':fault,'action':action,'y':y,'p':p,'oracle_a0':oracle(0,fault),'oracle_a1':oracle(1,fault),'oracle_f0':oracle(action,0),'oracle_f1':oracle(action,1)}

def run_synthetic(root,config,args):
    # Independent SEFN fit, risk fit, validation and test populations.
    fault_train,tr,va,te=[simulate(args.split_seed+100+i,args.scenario_rows) for i in range(4)]
    models=[]; effects=[]; faults=[]; predictions=[]; all_partitions=[]; synthetic_stats=[]
    for label,d in zip(['fault_train','risk_train','validation','test'],[fault_train,tr,va,te]):
        record=dict(partition=label,n=len(d['y']),positive=int(d['y'].sum()),negative=int(len(d['y'])-d['y'].sum()),fault_positive=int(d['fault'].sum()),fault_negative=int(len(d['fault'])-d['fault'].sum()))
        synthetic_stats.append(record); print('SYNTHETIC DATA | '+json.dumps(record),flush=True)
    atomic_csv(pd.DataFrame(synthetic_stats),root/'raw'/'synthetic_dataset_statistics.csv')
    for seed in config.seeds:
        seed_everything(seed)
        sefn,_,_=train_neural('ERM',fault_train['diag'],fault_train['fault'],fault_train['environment'],va['diag'],va['fault'],[2],['diagnostic'],seed,config,history_path=root/'runs'/f'synthetic_SEFN_{seed}'/'history.csv',run_id=f'synthetic_SEFN_{seed}')
        fprob=[predict_neural(sefn,d['diag'],torch.device(config.device)) for d in [tr,va,te]]; fth=choose_threshold(va['fault'],fprob[1]); fhat=[(p>=fth).astype(float) for p in fprob]
        faults.append(metrics_row(te['fault'],fprob[2],fth,dataset='synthetic',seed=seed,model='SEFN',split='independent_test',partition='test',calibration='raw'))
        faultrows=[]
        for part,d,p in [('train',fault_train,predict_neural(sefn,fault_train['diag'],torch.device(config.device))),('validation',va,fprob[1]),('test',te,fprob[2])]:
            faultrows.append(metrics_row(d['fault'],p,fth,dataset='synthetic',model='SEFN',seed=seed,partition=part,calibration='raw'))
        save_partition_metrics(root,f'synthetic_SEFN_{seed}',faultrows); all_partitions.extend(faultrows)
        state_net,_,_=train_neural('CIDT',tr['state'],tr['y'],tr['environment'],va['state'],va['y'],[tr['state'].shape[1]],['state'],seed,config,history_path=root/'runs'/f'synthetic_state_{seed}'/'history.csv',run_id=f'synthetic_state_{seed}')
        z=[predict_neural(state_net,d['state'],torch.device(config.device),return_representation=True)[1] for d in [tr,va,te]]
        def design(d,zz,h,a,include_fault=True):
            a=np.broadcast_to(a,(len(zz),)); h=np.broadcast_to(h,(len(zz),)); ss=np.eye(3)[d['scenario']]
            parts=[zz,a[:,None],ss[:,1:]*a[:,None]]
            if include_fault: parts.append(h[:,None])
            return np.concatenate(parts,axis=1).astype('float32')
        variants=['Full_SEFN_SCTM','Without_SEFN','Without_SCTM','Without_both','Oracle_fault_SCTM']
        for variant in variants:
            use_f=variant not in ['Without_SEFN','Without_both']; structural=variant in ['Full_SEFN_SCTM','Without_SEFN','Oracle_fault_SCTM']; hs=[d['fault'] for d in [tr,va,te]] if variant=='Oracle_fault_SCTM' else fhat
            mats=[design(d,zz,h,d['action'],use_f) for d,zz,h in zip([tr,va,te],z,hs)]
            if structural:
                model=LogisticRegression(max_iter=300,C=10,random_state=seed).fit(mats[0],tr['y']); predict=lambda a:model.predict_proba(a)[:,1]
            else:
                model,_,_=train_neural('ERM',mats[0],tr['y'],tr['environment'],mats[1],va['y'],[mats[0].shape[1]],['predictive'],seed,config,history_path=root/'runs'/f'synthetic_{variant}_{seed}'/'history.csv',run_id=f'synthetic_{variant}_{seed}'); predict=lambda a:predict_neural(model,a,torch.device(config.device))
            if structural: joblib.dump(model,root/'models'/f'synthetic_{variant}_{seed}.joblib')
            else: torch.save(model.state_dict(),root/'models'/f'synthetic_{variant}_{seed}.pt')
            pv,pt=predict(mats[1]),predict(mats[2]); ca=fit_cal(va['y'],pv); pcv=calibration_apply(ca,pv); pc=calibration_apply(ca,pt); th=choose_threshold(va['y'],pcv)
            ptr=predict(mats[0]); rawth=choose_threshold(va['y'],pv); variantrows=[]
            for part,d,raw in zip(['train','validation','test'],[tr,va,te],[ptr,pv,pt]):
                for kind,prob,threshold in [('raw',raw,rawth),('platt',calibration_apply(ca,raw),th)]:
                    variantrows.append(metrics_row(d['y'],prob,threshold,dataset='synthetic_controlled',model=variant,seed=seed,partition=part,calibration=kind))
            save_partition_metrics(root,f'synthetic_{variant}_{seed}',variantrows); all_partitions.extend(variantrows)
            atomic_csv(pd.DataFrame(all_partitions),root/'raw'/'synthetic_partition_metrics.csv')
            joblib.dump({'calibrator':ca,'threshold':th,'fault_threshold':fth},root/'models'/f'synthetic_{variant}_{seed}_settings.joblib')
            pa0=predict(design(te,z[2],hs[2],0,use_f)); pa1=predict(design(te,z[2],hs[2],1,use_f)); pf0=predict(design(te,z[2],0,te['action'],use_f)); pf1=predict(design(te,z[2],1,te['action'],use_f))
            for sc in [-1,0,1,2]:
                mask=np.ones(len(pc),bool) if sc==-1 else te['scenario']==sc; name='All' if sc==-1 else SCENARIOS[sc]
                if np.unique(te['y'][mask]).size!=2: continue
                models.append(metrics_row(te['y'][mask],pc[mask],th,dataset='synthetic_controlled',scenario=name,model=variant,seed=seed))
                for intervention,p0,p1,o0,o1 in [('action_0_to_1',pa0,pa1,te['oracle_a0'],te['oracle_a1']),('fault_0_to_1',pf0,pf1,te['oracle_f0'],te['oracle_f1'])]:
                    delta=(p1-p0)[mask]; truth=(o1-o0)[mask]; effects.append(dict(scenario=name,model=variant,seed=seed,intervention=intervention,estimated_ate=delta.mean(),oracle_ate=truth.mean(),absolute_ate_error=abs(delta.mean()-truth.mean()),individual_effect_mae=np.abs(delta-truth).mean(),interpretation='SCM readout' if structural else 'predictive perturbation'))
            predictions.append(pd.DataFrame({'seed':seed,'model':variant,'scenario':te['scenario'],'row':np.arange(len(pc)),'target':te['y'],'fault':te['fault'],'action':te['action'],'fault_estimate':hs[2],'predicted_risk':pc,'threshold':th,'oracle_risk':te['p']}))
        torch.save(sefn.state_dict(),root/'models'/f'synthetic_SEFN_{seed}.pt'); torch.save(state_net.state_dict(),root/'models'/f'synthetic_state_{seed}.pt')
    atomic_csv(pd.concat(predictions),root/'raw'/'synthetic_predictions.csv.gz'); atomic_csv(pd.DataFrame(models),root/'raw'/'synthetic_metrics.csv'); atomic_csv(pd.DataFrame(effects),root/'raw'/'synthetic_interventions.csv'); atomic_csv(pd.DataFrame(faults),root/'raw'/'synthetic_sefn.csv')
    json_write(root/'raw'/'synthetic_protocol.json',{'status':'controlled synthetic ground-truth SCM, not physical driving simulator or real-world decision validation','scenarios':SCENARIOS,'partitions':['SEFN training','risk training','validation','test'],'rows_per_partition':args.scenario_rows,'action':'randomized protective action','fault':'Bernoulli conditional on context and environment','graph':'context/environment -> fault; fault/context -> diagnostic; scenario/hazard/context/fault/action -> outcome; diagnostic -> estimated fault; state -> latent representation','oracle':'sigmoid(-0.7 + 0.95 hazard + 0.35 context + 1.25 fault + scenario_specific_action_coefficient * action + 0.2 I(scenario=V2X))','action_coefficients':[-1.65,-1.2,-1.0]})
    return pd.DataFrame(models),pd.DataFrame(effects),pd.DataFrame(faults)

METRICS=['auroc','auprc','brier','ece','adaptive_ece','nll','accuracy','balanced_accuracy','mcc','f1','fsr','fur','threshold','unsafe_prediction_rate','prevalence','model_size_mib','latency_median_ms','latency_p95_ms','training_seconds','best_epoch','epochs_completed','selected_epoch']

def plot_bars(root,name,df,key='model',metric='auroc_mean',ylabel='AUROC'):
    if df.empty: return
    fig,ax=plt.subplots(figsize=(max(6,min(13,len(df)*.45)),4)); x=np.arange(len(df)); sd=metric.replace('_mean','_sd'); err=df[sd].fillna(0) if sd in df else None
    ax.bar(x,df[metric],yerr=err,capsize=3); ax.set_xticks(x,df[key].astype(str),rotation=30,ha='right'); ax.set_ylabel(ylabel); ax.grid(axis='y',alpha=.25)
    if 'auroc' in metric: ax.axhline(.5,ls='--',color='gray',lw=1); ax.set_ylim(0,1.05)
    save_plot(root,name,fig)

def report(root,result,stats,manifest,syn,effects,faults,args):
    data={k:pd.DataFrame(v) for k,v in result.items()}; m=data['metrics']; d=data['deployment']; s=data['sensitivity']; b=data['bootstrap']; mf=pd.DataFrame(manifest)
    for k,v in data.items(): atomic_csv(v,root/'raw'/(k+'.csv'))
    if len(m):
        publish_table(root,'Table_S12_all_partition_metrics',seed_summary(m,['dataset','split','model','partition','calibration'],METRICS),'Train, validation and test metrics; final neural checkpoint is the last completed epoch.',latex=False)
        publish_table(root,'Table_S13_all_partition_per_seed',m,'Full per-seed train, validation and test metrics, raw and calibrated.',latex=False)
        test=m[(m.partition=='test')&(m.calibration=='platt')].copy(); summ=seed_summary(test,['dataset','split','model'],METRICS)
        primary=summ[~summ.model.str.startswith(('CIDT_no_','CIDT_fraction_','IRM_lambda_'))]
        core=summ[summ.apply(lambda r:r['split']==CORE[r.dataset],axis=1)]
        loss=core[core.model.isin(['CIDT','ERM']+ABLATIONS)]; columns=['dataset','split','model','runs','auroc_mean','auroc_sd','auprc_mean','brier_mean','ece_mean','fsr_mean','fur_mean','balanced_accuracy_mean']
        publish_table(root,'Table_04_IID_OOD',primary,'Replacement main results: validation-calibrated test results (mean and sample SD across seeds). Different tasks must not be pooled.',columns)
        iid=primary[primary.split=='iid'].copy()
        train=m[(m.partition=='train')&(m.calibration=='raw')&(m.split=='iid')].groupby(['dataset','model'],as_index=False).auroc.mean().rename(columns={'auroc':'train_auroc_mean'})
        iid=iid.merge(train,on=['dataset','model'],how='left')
        publish_table(root,'Table_04a_IID_predictive',iid,'IID prediction; train AUROC raw, test AUROC and AUPRC validation-Platt; sample SD and Student-t CI.', ['dataset','model','runs','train_auroc_mean','auroc_mean','auroc_sd','auroc_t95_low','auroc_t95_high','auprc_mean','auprc_sd','auprc_t95_low','auprc_t95_high'])
        publish_table(root,'Table_04b_IID_calibration_safety',iid,'IID validation-Platt calibration, safety and CPU model-only measurements.', ['dataset','model','runs','brier_mean','ece_mean','adaptive_ece_mean','nll_mean','fsr_mean','fur_mean','balanced_accuracy_mean','mcc_mean','latency_median_ms_mean','model_size_mib_mean'])
        rawtest=m[(m.partition=='test')&(m.calibration=='raw')]; publish_table(root,'Table_S11_raw_test_metrics',seed_summary(rawtest,['dataset','split','model'],METRICS),'Uncalibrated test metrics, useful for identifying any calibration-induced ranking reversal.',latex=False)
        # Reliability plot: aggregate fixed-test seed curves by bin, retaining raw and calibrated predictions.
        fig,axes=plt.subplots(1,2,figsize=(10,4)); used=False
        for ax,ds in zip(axes,['dabra','veremi']):
            ax.plot([0,1],[0,1],ls=':',color='gray'); ax.set(xlabel='Mean predicted probability',ylabel='Observed positive fraction',title=ds.upper(),xlim=(0,1),ylim=(0,1))
            for kind in ['raw','calibrated']:
                bins=[]
                for seed in args.seeds:
                    pp=root/'predictions'/f'{ds}__iid__CIDT__s{seed}.csv.gz'
                    if not pp.exists(): continue
                    df=pd.read_csv(pp,usecols=['target',kind+'_probability']); pvals=df[kind+'_probability'].to_numpy(); bid=np.minimum(14,np.floor(pvals*15).astype(int)); g=df.assign(bin=bid).groupby('bin').agg(pp=(kind+'_probability','mean'),yy=('target','mean')); bins.append(g)
                if bins:
                    g=pd.concat(bins).groupby(level=0).mean(); ax.plot(g.pp,g.yy,marker='o',label=kind); used=True
            if ax.get_legend_handles_labels()[0]: ax.legend()
        if used: save_plot(root,'Figure_S02_reliability',fig)
        else: plt.close(fig)
        for ds,suffix in [('dabra','a_DABRA'),('veremi','b_VeReMi')]:
            g=primary[(primary.dataset==ds)&primary.model.isin(['CIDT','ERM','IRM','SCTM_latent_readout'])]
            if not g.empty:
                fig,axes=plt.subplots(2,2,figsize=(10,7)); order=list(g.split.unique())
                for ax,metric in zip(axes.flat,['auroc','auprc','brier','ece']):
                    for mod,gg in g.groupby('model'):
                        gg=gg.set_index('split').reindex(order); ax.errorbar(range(len(order)),gg[metric+'_mean'],yerr=gg[metric+'_sd'].fillna(0),marker='o',capsize=2,label=mod)
                    ax.set_xticks(range(len(order)),order,rotation=25,ha='right'); ax.set_title(metric.upper()); ax.grid(alpha=.2)
                    if metric=='auroc': ax.axhline(.5,ls=':',color='gray')
                axes[0,0].legend(fontsize=7); save_plot(root,'Figure_03'+suffix+'_OOD',fig)
            g=loss[loss.dataset==ds]
            if len(g):
                eff=core[(core.dataset==ds)&(core.model.str.startswith('CIDT_fraction_')|core.model.eq('CIDT'))].copy(); eff['fraction']=eff.model.map(lambda v:1. if v=='CIDT' else float(v.split('_')[-1])); eff=eff.sort_values('fraction')
                fig,axes=plt.subplots(1,2,figsize=(11,4.5)); ax=axes[0]; ax.errorbar(eff.fraction,eff.auroc_mean,yerr=eff.auroc_sd.fillna(0),marker='o',label='AUROC'); ax.axhline(.5,ls=':',color='gray'); ax.set(xlabel='Fraction of fixed training pool',ylabel='OOD AUROC',ylim=(0,1.05)); twin=ax.twinx(); twin.plot(eff.fraction,eff.best_epoch_mean,marker='s',ls='--',color='tab:orange'); twin.set_ylabel('Best validation epoch'); ax=axes[1]; ax.bar(np.arange(len(g)),g.auroc_mean,yerr=g.auroc_sd.fillna(0),capsize=2); ax.set_xticks(np.arange(len(g)),g.model,rotation=35,ha='right'); ax.axhline(.5,ls=':',color='gray'); ax.set(ylabel='OOD AUROC',ylim=(0,1.05)); save_plot(root,'Figure_04'+suffix+'_ablation',fig)
        publish_table(root,'Table_05_loss_ablation',loss,'Objective ablations on the exact held-out split used in Figure 3; SCTM is post-hoc and is not a causal loss.',columns)
        baseline=test[test.model=='CIDT'][['dataset','split','seed','auroc']].rename(columns={'auroc':'full_auroc'}); paired=test[test.model.isin(ABLATIONS+['ERM'])].merge(baseline,on=['dataset','split','seed']); paired=paired[paired.apply(lambda r:r.split==CORE[r.dataset],axis=1)]; paired['delta_auroc']=paired.auroc-paired.full_auroc
        if len(paired): publish_table(root,'Table_S03_paired_ablation_deltas',seed_summary(paired,['dataset','split','model'],['delta_auroc']),'Paired difference (ablation minus full model), same test rows and seeds.')
        ir=core[core.model.str.startswith('IRM_lambda_')]; publish_table(root,'Table_09_IRM_sensitivity',ir,'Fixed IRM penalty sweep; no predetermined optimal value.',columns)
        publish_table(root,'Table_S05_per_seed_metrics',m,'All raw and Platt metrics, including training and validation results.',latex=False); publish_table(root,'Table_S06_uncertainty',summ,'Per-model sample SD and two-sided Student-t interval across training seeds conditional on fixed split.',latex=False)
        publish_table(root,'Table_S09_standalone_module_results',primary[(primary.model.isin(['CIDT','SCTM_latent_readout']))],'DABRA CIDT risk and VeReMi CIDT/SEFN fault task, plus reduced SCTM latent readout. Not integrated cross-dataset evidence.',columns)
        # Figure 2 replacement uses actual realized split class counts.
        if len(mf):
            g=mf[(mf.status=='complete')&(mf.split=='iid')].copy(); g['label']=g.dataset+' '+g.partition
            fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(len(g)); ax.bar(x,g.n-g.positive,label='Negative'); ax.bar(x,g.positive,bottom=g.n-g.positive,label='Positive'); ax.set_xticks(x,g.label,rotation=25,ha='right'); ax.set_ylabel('Rows'); ax.legend(); save_plot(root,'Figure_02_dataset_structure',fig)
    if len(mf): publish_table(root,'Table_S01_split_manifest',mf,'Exact data counts, prevalence and held-out groups.',latex=False)
    if stats: publish_table(root,'Table_03_dataset_protocol',pd.DataFrame(stats).assign(train_fraction_requested=args.train_fraction,validation_fraction_requested=.1,test_fraction_requested=.1,training_seeds=','.join(map(str,args.seeds)),split_seed=args.split_seed),'Source and analysis counts; exact realized partitions in supplementary split manifest.')
    legacy=pd.DataFrame([['DABRA','Figure 3',.73,.71,.68],['VeReMi','Figure 3',.98,.95,.90],['DABRA','Figure 4',.4956,np.nan,np.nan],['VeReMi','Figure 4',.712,np.nan,np.nan]],columns=['dataset','old_source','reported_1','reported_2','reported_3']); legacy['provenance']='Reviewer quotation only; not independently verified; do not combine with new rerun'; publish_table(root,'Table_S02_legacy_OOD_values',legacy,'Old reported values retained only for discrepancy audit.')
    if len(b): publish_table(root,'Table_S07_group_bootstrap',b,'Cluster bootstrap conditional on fitted model, driver or sender grouping; row cap and valid replicate counts reported. Few groups imply limited precision.',latex=False)
    if len(s):
        ss=seed_summary(s,['dataset','split','feature'],['baseline_risk','perturbed_risk','mean_risk_change','mean_absolute_change','auroc_change']); publish_table(root,'Table_S04_feature_sensitivity',ss,'Training-mean replacement sensitivity. Not identified causal effects.', ['dataset','split','feature','runs','mean_risk_change_mean','mean_risk_change_sd','mean_absolute_change_mean','auroc_change_mean'])
        fig,axes=plt.subplots(1,2,figsize=(11,4.5));
        for ax,ds in zip(axes,['dabra','veremi']):
            g=ss[ss.dataset==ds]; ax.bar(range(len(g)),g.mean_risk_change_mean,yerr=g.mean_risk_change_sd.fillna(0),capsize=2); ax.set_xticks(range(len(g)),g.feature,rotation=35,ha='right'); ax.axhline(0,color='gray',lw=1); ax.set_title(ds.upper()); ax.set_ylabel('Mean change in predicted probability')
        save_plot(root,'Figure_05b_feature_sensitivity',fig)
    if len(d):
        ds=seed_summary(d,['dataset','split','precision','setting'],METRICS+['throughput_single_sample_per_s']); publish_table(root,'Table_06_deployment',ds,'Matched CPU dynamic INT8 vs FP32; shared raw and shared Platt operating points are primary, own validation calibration is secondary.', ['dataset','split','precision','setting','runs','auroc_mean','fsr_mean','fur_mean','threshold_mean','balanced_accuracy_mean','model_size_mib_mean','latency_median_ms_mean','latency_p95_ms_mean'])
        dpcols=['auroc','auprc','brier','fsr','fur','balanced_accuracy','model_size_mib','latency_median_ms']; pp=d.pivot(index=['dataset','split','seed','setting'],columns='precision',values=dpcols); delta=pd.DataFrame({k:pp[k]['INT8']-pp[k]['FP32'] for k in dpcols}).reset_index(); publish_table(root,'Table_S08_deployment_paired',seed_summary(delta,['dataset','split','setting'],dpcols),'Paired INT8 minus FP32 differences.',latex=False)
        g=ds[ds.setting=='shared_platt'].copy(); g['label']=g.dataset+'\n'+g.split+'\n'+g.precision
        fig,axes=plt.subplots(2,2,figsize=(12,7));
        for ax,k in zip(axes.flat,['model_size_mib','latency_median_ms','auroc','balanced_accuracy']): ax.bar(range(len(g)),g[k+'_mean'],yerr=g[k+'_sd'].fillna(0),capsize=2); ax.set_xticks(range(len(g)),g.label,rotation=35,ha='right',fontsize=7); ax.set_title(k)
        save_plot(root,'Figure_06_deployment',fig)
        fig,ax=plt.subplots(figsize=(10,4)); x=np.arange(len(g)); ax.bar(x-.18,g.fsr_mean,.36,label='FSR'); ax.bar(x+.18,g.fur_mean,.36,label='FUR'); ax.set_xticks(x,g.label,rotation=30,ha='right'); ax.set_ylim(0,1.05); ax.legend(); save_plot(root,'Figure_S01_FSR_FUR',fig)
    if len(syn):
        sm=seed_summary(syn,['scenario','model'],METRICS); es=seed_summary(effects,['scenario','model','intervention'],['estimated_ate','oracle_ate','absolute_ate_error','individual_effect_mae']); publish_table(root,'Table_07_scenario_component_ablation',sm,'Synthetic randomized-action scenario benchmark only; not physical simulation or real driving evidence.', ['scenario','model','runs','auroc_mean','auroc_sd','auprc_mean','brier_mean','fsr_mean','fur_mean','balanced_accuracy_mean']); publish_table(root,'Table_08_scenario_interventions',es,'Estimated vs known synthetic SCM effects; estimated ATE is not real-world causal evidence.', ['scenario','model','intervention','runs','oracle_ate_mean','estimated_ate_mean','estimated_ate_sd','absolute_ate_error_mean','individual_effect_mae_mean']); publish_table(root,'Table_S10_synthetic_SEFN',seed_summary(faults,['dataset','model'],METRICS),'SEFN independent synthetic fault test.', ['dataset','model','runs','auroc_mean','auroc_sd','auprc_mean','fsr_mean','fur_mean'])
        plot_bars(root,'Figure_07_scenarios',sm[sm.scenario=='All'])
        g=es[(es.model=='Full_SEFN_SCTM')&(es.scenario!='All')]; fig,ax=plt.subplots(figsize=(10,4)); x=np.arange(len(g)); ax.bar(x-.18,g.oracle_ate_mean,.36,label='Oracle'); ax.bar(x+.18,g.estimated_ate_mean,.36,yerr=g.estimated_ate_sd.fillna(0),label='Estimated'); ax.set_xticks(x,g.scenario+'\n'+g.intervention,rotation=25,ha='right'); ax.axhline(0,color='gray'); ax.set_ylabel('Average intervention effect'); ax.legend(); save_plot(root,'Figure_08_interventions',fig)
        fig,ax=plt.subplots(figsize=(9,4)); ax.axis('off'); pos={'Environment':(.10,.78),'Context':(.10,.50),'Fault F':(.43,.78),'Diagnostic':(.73,.78),'Hazard / scenario':(.18,.15),'Action A':(.50,.15),'Safety Y':(.86,.36)}
        for a,(x,y) in pos.items(): ax.text(x,y,a,ha='center',va='center',bbox=dict(boxstyle='round',fc='white',ec='black'),transform=ax.transAxes)
        for a,c in [('Environment','Fault F'),('Context','Fault F'),('Fault F','Diagnostic'),('Context','Diagnostic'),('Fault F','Safety Y'),('Context','Safety Y'),('Hazard / scenario','Safety Y'),('Action A','Safety Y')]: ax.annotate('',xy=pos[c],xytext=pos[a],xycoords='axes fraction',textcoords='axes fraction',arrowprops=dict(arrowstyle='->',shrinkA=32,shrinkB=30,connectionstyle='arc3,rad=0.08'))
        ax.set_title('Specified synthetic causal graph (not discovered from observed datasets)'); save_plot(root,'Figure_05a_causal_graph',fig)
    distribute(root,args)

def distribute(root,args):
    notes={
    'R1_C01_OOD_consistency':'Compare refreshed Figure 3 and Figure 4 CIDT results on the same held-out split. Legacy figures cannot be reconciled without original prediction provenance; replace rather than invent an explanation.',
    'R1_C02_objective_ablation':'No_IRM removes IRM only, no_CORAL removes feature alignment only, no_group_variance removes environment-risk variance only. No_auxiliary retains group-averaged empirical risk; ERM uses pooled BCE. No separate causal training loss is present in the manuscript equations.',
    'R1_C03_VeReMi_sensitivity':'Feature blocks are position/speed/acceleration/heading (x,y and associated noise columns); mean replacement is sensitivity only.',
    'R1_C04_seeds_uncertainty':'Fixed splits, repeated training seeds; sample SD and Student-t intervals measure training variability conditional on split. Cluster bootstrap describes conditional test uncertainty; not interchangeable. See split manifests and replicate counts.',
    'R1_C05_FP32_INT8':'Compare shared_raw and shared_platt first. High FUR must be acknowledged, even at low FSR. Latency is CPU forward only, not end-to-end driving.',
    'R1_C06_integrated_SEFN':'Real datasets are independent tasks. Synthetic benchmark is a limited controlled integration demonstration with known action and fault assignments. Do not claim actual AV decision verification.',
    'R2_C01_OOD_consistency':'Same as R1 C01; compare only identical protocols.',
    'R2_C02_component_ablation':'Real-data latent SCTM is observational readout; synthetic controlled benchmark compares explicit SEFN removal and replacing structural readout by predictive MLP.',
    'R2_C03_driving_decisions':'Three synthetic action scenarios, not a vehicle dynamics simulator. External physical simulation or genuine aligned action/fault/outcome dataset is still required for real-world claims.',
    'R2_C04_causal_validation':'Synthetic graph and oracle equations are pre-specified; real feature replacement is not causal identification. Discuss assumptions and limitations.',
    'R3_C01_quantization':'Dynamic post-training INT8 on Linear layers, CPU only, no INT8 training. Report exact measured predictive deltas and serialized state-dict MiB.'}
    index=[]
    for comment,stems in REVIEW.items():
        folder=root/'reviewers'/comment; folder.mkdir(parents=True,exist_ok=True); found=[]; missing=[]
        for stem in stems:
            files=list((root/'paper'/'tables').glob(stem+'.*'))+list((root/'paper'/'tables').glob(stem+'_caption.txt'))+list((root/'paper'/'figures').glob(stem+'.*'))
            if not files: missing.append(stem)
            for file in files: shutil.copy2(file,folder/file.name); found.append(file.name)
        (folder/'READ_ME.txt').write_text(notes[comment]+'\n\nGenerated files:\n'+'\n'.join(found)+'\n\nUnavailable/not run:\n'+'\n'.join(missing),encoding='utf-8'); index.append({'comment':comment,'folder':str(folder.relative_to(root)),'generated':'; '.join(found),'missing':'; '.join(missing),'interpretation':notes[comment]})
    atomic_csv(pd.DataFrame(index),root/'reviewer_index.csv')
    text=['# Manuscript replacement map','Do not copy old numerical claims. Use the generated CSV/LaTeX tables and figure captions. Main paper numbering below is proposed; LaTeX will determine final numbering.','',f'Training seeds: {args.seeds}; neural epochs: {args.epochs} (last epoch, no early stopping); split seed: {args.split_seed}; requested train/validation/test fractions: {args.train_fraction}/0.10/0.10; exact sizes in Table_S01.','', 'Existing Table 3 -> Table_03_dataset_protocol plus split manifest.', 'Existing Table 4 -> Table_04_IID_OOD (full CSV includes all metrics and uncertainty).', 'Existing Fig 2 -> Figure_02_dataset_structure.', 'Existing Fig 3 -> Figure_03a_DABRA_OOD and Figure_03b_VeReMi_OOD.', 'Existing Fig 4 -> Figure_04a_DABRA_ablation and Figure_04b_VeReMi_ablation (same held-out CIDT point as Fig 3).', 'Existing Fig 5 -> Figure_05a_causal_graph (synthetic only) and Figure_05b_feature_sensitivity (observational only).', 'Existing Fig 6 -> Figure_06_deployment (direct measured metrics; replace radar).', 'New Table 5 loss ablation; Table 6 deployment; Tables 7–8 synthetic components and intervention oracle; Table 9 IRM sweep (old IRM table).', 'New Fig 7–8 controlled scenario and oracle effects.', 'Figure 1 architecture, literature and notation tables may be retained only where consistent; dataset counts must reflect actual split.','', '### Scope and interpretation', 'Confine DABRA to derived risk classification and VeReMi to message fault detection. Synthetic results are not CARLA/SUMO or real-world evidence. No claims of identified observational causal effects. Do not claim auxiliary terms improve performance unless paired estimates support it. Report near-chance AUROC and high FUR plainly. If a split is unavailable, report why; do not substitute another split silently.']
    (root/'paper'/'MANUSCRIPT_UPDATE_MAP.md').write_text('\n'.join(text),encoding='utf-8')


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log
    def write(self, text):
        self.terminal.write(text); self.log.write(text); self.log.flush()
        return len(text)
    def flush(self):
        self.terminal.flush(); self.log.flush()
    def __getattr__(self, key):
        return getattr(self.terminal, key)


def main():
    parser=argparse.ArgumentParser(description='CIDT reproducible reviewer revision runner')
    parser.add_argument('--dabra',default=r'E:\other\Causal-DT\Dataset\DABRA.csv'); parser.add_argument('--veremi',default=r'E:\other\Causal-DT\Dataset\VeReMi.csv'); parser.add_argument('--out',default=r'E:\other\Causal-DT\Reviewer_Revision_Results_50Epochs')
    parser.add_argument('--dataset',choices=['both','dabra','veremi','synthetic'],default='both'); parser.add_argument('--profile',choices=['paper','smoke'],default='paper'); parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto'); parser.add_argument('--workers',type=int,default=4); parser.add_argument('--seeds',type=int,nargs='+',default=SEEDS); parser.add_argument('--split-seed',type=int,default=11); parser.add_argument('--train-fraction',type=float,default=.50); parser.add_argument('--epochs',type=int,default=50); parser.add_argument('--batch-size',type=int,default=2048); parser.add_argument('--bootstrap',type=int,default=500); parser.add_argument('--dabra-limit',type=int,default=0); parser.add_argument('--veremi-limit',type=int,default=0); parser.add_argument('--scenario-rows',type=int,default=12000); parser.add_argument('--latency-repeats',type=int,default=200); parser.add_argument('--skip-synthetic',action='store_true'); parser.add_argument('--no-resume',action='store_true'); args=parser.parse_args()
    if not .1<=args.train_fraction<=.8 or args.workers<1 or args.epochs<1 or args.batch_size<2 or args.scenario_rows<100 or not args.seeds or len(set(args.seeds))!=len(args.seeds): parser.error('Invalid fraction, workers, epochs, batch size, scenario rows or duplicate seeds')
    if args.profile=='paper' and args.epochs<50: parser.error('Paper runs require at least 50 epochs; use smoke for short tests')
    if args.profile=='smoke': args.seeds=args.seeds[:1]; args.epochs=min(args.epochs,2); args.bootstrap=min(args.bootstrap,5); args.dabra_limit=args.dabra_limit or 4000; args.veremi_limit=args.veremi_limit or 12000; args.scenario_rows=min(args.scenario_rows,800); args.latency_repeats=min(args.latency_repeats,20)
    root=Path(args.out); root.mkdir(parents=True,exist_ok=True)
    selected=['dabra','veremi'] if args.dataset=='both' else ([] if args.dataset=='synthetic' else [args.dataset])
    for ds in selected:
        if not Path(getattr(args,ds)).is_file(): parser.error(f'{ds} dataset not found: {getattr(args,ds)}')
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    if device=='cuda' and not torch.cuda.is_available(): parser.error('CUDA requested but unavailable')
    torch.set_num_threads(args.workers)
    signature={k:v for k,v in vars(args).items() if k not in ['no_resume','out']}; signature['resolved_device']=device; signature['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(); signature['inputs']={ds:{'path':str(Path(getattr(args,ds)).resolve()),'size':Path(getattr(args,ds)).stat().st_size,'mtime_ns':Path(getattr(args,ds)).stat().st_mtime_ns} for ds in selected}
    lock=root/'run_config.json'
    if lock.exists() and json.loads(lock.read_text(encoding='utf-8'))['signature']!=signature: parser.error('Output directory has a different run configuration. Use a new output folder; prevents mixing incompatible results.')
    json_write(lock,{'signature':signature,'platform':platform.platform(),'processor':platform.processor(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'smoke_test_not_for_publication':args.profile=='smoke'})
    for dd in ['raw','models','runs','predictions','paper/tables','paper/figures','splits','reviewers']: (root/dd).mkdir(parents=True,exist_ok=True)
    config=RunConfig(args.profile,1.,args.seeds,args.epochs,args.batch_size,args.bootstrap,args.veremi_limit,args.workers,device)
    logpath=root/'logs'/('run_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.log')
    logpath.parent.mkdir(parents=True,exist_ok=True)
    with logpath.open('a',encoding='utf-8',buffering=1) as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout,logfile)), contextlib.redirect_stderr(Tee(sys.stderr,logfile)):
            print(f'RUN START | {datetime.datetime.now().isoformat()} | device={device} | epochs={args.epochs} | early_stopping=OFF | output={root}',flush=True)
            print('CONFIG | '+json.dumps(signature,default=str),flush=True)
            try:
                result={k:[] for k in ['metrics','bootstrap','deployment','sensitivity','components']}; stats=[]; manifest=[]; syn=pd.DataFrame(); effects=pd.DataFrame(); faults=pd.DataFrame()
                for ds in selected:
                    rr,st,mf=run_real(root,ds,Path(getattr(args,ds)),config,args); stats.append(st); manifest.extend(mf)
                    for k in result: result[k].extend(rr[k])
                    report(root,result,stats,manifest,syn,effects,faults,args)
                if not args.skip_synthetic: syn,effects,faults=run_synthetic(root,config,args)
                report(root,result,stats,manifest,syn,effects,faults,args)
                if args.profile=='smoke': (root/'SMOKE_TEST_NOT_FOR_PUBLICATION.txt').write_text('Reduced one-seed test; not manuscript evidence.',encoding='utf-8')
                print('Completed:',root,flush=True)
            except BaseException:
                traceback.print_exc()
                raise


if __name__=='__main__': main()
