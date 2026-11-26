#!/usr/bin/env python
"""Reviewer experiment: deployment-only FP32 versus dynamic INT8 quantization."""

from pathlib import Path
import json, math, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, accuracy_score, balanced_accuracy_score, matthews_corrcoef, f1_score,
    precision_score, recall_score)
warnings.filterwarnings("ignore")

ROOT=Path(r"E:\other\Causal-DT")
DATA=ROOT/"Dataset"
FINAL=ROOT/"FinalResults_50_10_10"
OUTROOT=FINAL/"reviewer_extra_experiments"
SEEDS=[5,11,23,37,45]

def snake(x):
    import re
    s=str(x).strip()
    s=re.sub(r"([a-z0-9])([A-Z])",r"\1_\2",s)
    return re.sub(r"[^A-Za-z0-9]+","_",s).strip("_").lower()

def ece(y,p,bins=15):
    y=np.asarray(y); p=np.asarray(p); ans=0.
    ed=np.linspace(0,1,bins+1)
    for i in range(bins):
        m=(p>=ed[i]) & (p <= ed[i+1] if i==bins-1 else p<ed[i+1])
        if m.any(): ans += m.mean()*abs(y[m].mean()-p[m].mean())
    return float(ans)

def aece(y,p,bins=15):
    y=np.asarray(y); p=np.asarray(p); order=np.argsort(p); ans=0.
    for idx in np.array_split(order,bins):
        if len(idx): ans += len(idx)/len(y)*abs(y[idx].mean()-p[idx].mean())
    return float(ans)

def choose_threshold(y,p,max_fsr=.05):
    best=None
    for th in np.unique(np.r_[np.linspace(.01,.99,199),np.quantile(p,np.linspace(.01,.99,99))]):
        pred=(p>=th).astype(int); pos=y==1; neg=y==0
        fsr=((pred==0)&pos).sum()/max(1,pos.sum())
        fur=((pred==1)&neg).sum()/max(1,neg.sum())
        ba=balanced_accuracy_score(y,pred)
        key=(int(fsr<=max_fsr),ba,-fur,-th)
        if best is None or key>best[0]: best=(key,float(th))
    return best[1]

def metrics(y,p,th):
    p=np.clip(np.asarray(p),1e-7,1-1e-7); y=np.asarray(y,dtype=int)
    pred=(p>=th).astype(int); pos=y==1; neg=y==0
    return dict(
        AUROC=roc_auc_score(y,p), AUPRC=average_precision_score(y,p),
        Brier=brier_score_loss(y,p), ECE=ece(y,p), AECE=aece(y,p),
        NLL=log_loss(y,p,labels=[0,1]), Accuracy=accuracy_score(y,pred),
        BAcc=balanced_accuracy_score(y,pred), MCC=matthews_corrcoef(y,pred),
        F1=f1_score(y,pred,zero_division=0), Precision=precision_score(y,pred,zero_division=0),
        Recall=recall_score(y,pred,zero_division=0),
        FSR=((pred==0)&pos).sum()/max(1,pos.sum()),
        FUR=((pred==1)&neg).sum()/max(1,neg.sum()), Threshold=th,
        Prevalence=y.mean()
    )

def split_idx(y,seed,train=.20,test=.10):
    rng=np.random.default_rng(seed); tr=[]; te=[]
    for c in np.unique(y):
        z=np.flatnonzero(y==c); rng.shuffle(z)
        a=int(round(train*len(z))); b=int(round(test*len(z)))
        tr.extend(z[:a]); te.extend(z[a:a+b])
    return np.asarray(tr),np.asarray(te)

def fit_val(idx,y,seed):
    rng=np.random.default_rng(seed+9000); fit=[]; val=[]
    for c in np.unique(y[idx]):
        loc=np.flatnonzero(y[idx]==c); rng.shuffle(loc); n=max(1,int(.2*len(loc)))
        val.extend(idx[loc[:n]]); fit.extend(idx[loc[n:]])
    return np.asarray(fit),np.asarray(val)

def pipe_for(df,cols):
    cols=[c for c in cols if c in df.columns]
    nums=[c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cats=[c for c in cols if c not in nums]; trans=[]
    if nums: trans.append(("n",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),nums))
    if cats: trans.append(("c",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),cats))
    return Pipeline([("pre",ColumnTransformer(trans)),("lr",LogisticRegression(solver="liblinear",max_iter=300))]),cols

def summarize(raw,groupcols):
    rows=[]
    for key,g in raw.groupby(groupcols,sort=False):
        if not isinstance(key,tuple): key=(key,)
        r=dict(zip(groupcols,key)); r["Runs"]=g["Seed"].nunique()
        for m in [c for c in raw.columns if c not in groupcols+["Seed"] and pd.api.types.is_numeric_dtype(raw[c])]:
            x=pd.to_numeric(g[m],errors="coerce").dropna()
            if not len(x): continue
            r[m+"_mean"]=x.mean(); r[m+"_sd"]=x.std(ddof=1)
            if len(x)>1:
                q=t.ppf(.975,len(x)-1)*x.std(ddof=1)/math.sqrt(len(x))
                r[m+"_ci_low"]=x.mean()-q; r[m+"_ci_high"]=x.mean()+q
        rows.append(r)
    return pd.DataFrame(rows)

def load_dabra():
    p=FINAL/"data"/"dabra_full_with_target.parquet"
    if p.exists(): df=pd.read_parquet(p)
    else: df=pd.read_csv(DATA/"DABRA.csv",low_memory=False)
    df.columns=[snake(c) for c in df.columns]
    tc=next((c for c in ["target","derived_failure_tau_3"] if c in df.columns),None)
    if tc is None:
        req=["anomalous_event","route_anomaly","geofencing_violation","stop_events"]
        if any(c not in df.columns for c in req): raise ValueError("DABRA target unavailable")
        y=((pd.to_numeric(df.anomalous_event,errors="coerce").fillna(0)==1)|
           (pd.to_numeric(df.route_anomaly,errors="coerce").fillna(0)==1)|
           (pd.to_numeric(df.geofencing_violation,errors="coerce").fillna(0)==1)|
           (pd.to_numeric(df.stop_events,errors="coerce").fillna(0)>=3)).astype(int).to_numpy()
    else: y=pd.to_numeric(df[tc],errors="raise").astype(int).to_numpy()
    return df,y

def load_veremi():
    p=DATA/"VeReMi.csv"; hdr=pd.read_csv(p,nrows=0).columns
    mp={snake(c):c for c in hdr}
    groups={"position":["posx","posy","posx_n","posy_n"],"speed":["spdx","spdy","spdx_n","spdy_n"],
    "acceleration":["aclx","acly","aclx_n","acly_n"],"heading":["hedx","hedy","hedx_n","hedy_n"]}
    need=["class"]+sum(groups.values(),[])
    raw=[mp[c] for c in need if c in mp]; df=pd.read_csv(p,usecols=raw,low_memory=False); df.columns=[snake(c) for c in df.columns]
    y=(pd.to_numeric(df["class"],errors="raise").astype(int)!=0).astype(int).to_numpy()
    return df,y,groups

import os, tempfile, copy
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class Net(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,64),nn.ReLU(),
                               nn.Linear(64,128),nn.ReLU(),nn.Dropout(.15),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze(1)

def size_mb(model):
    with tempfile.NamedTemporaryFile(suffix=".pt",delete=False) as f: path=f.name
    try:
        torch.save(model.state_dict(),path); return os.path.getsize(path)/(1024**2)
    finally:
        if os.path.exists(path): os.remove(path)

def infer(model,x,repeats=5):
    model.eval(); times=[]; out=None
    with torch.no_grad():
        for _ in range(repeats):
            t0=time.perf_counter(); z=model(x); times.append(time.perf_counter()-t0); out=torch.sigmoid(z).cpu().numpy()
    return out,1000*np.median(times)/len(x),1000*np.quantile(times,.95)/len(x)

def main():
    out=OUTROOT/"int8_deployment"; out.mkdir(parents=True,exist_ok=True)
    df,y,groups=load_veremi(); cols=[c for c in sum(groups.values(),[]) if c in df.columns]
    X=df[cols].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(dtype=np.float32); rows=[]
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed); tr,te=split_idx(y,seed); fit,val=fit_val(tr,y,seed)
        sc=StandardScaler().fit(X[fit]); xf=torch.from_numpy(sc.transform(X[fit]).astype("float32")); xv=torch.from_numpy(sc.transform(X[val]).astype("float32")); xt=torch.from_numpy(sc.transform(X[te]).astype("float32"))
        net=Net(len(cols)); opt=torch.optim.Adam(net.parameters(),lr=1e-3); loss=nn.BCEWithLogitsLoss()
        net.train()
        for ep in range(12):
            perm=torch.randperm(len(xf))
            for s in range(0,len(xf),4096):
                ix=perm[s:s+4096]; opt.zero_grad(); l=loss(net(xf[ix]),torch.from_numpy(y[fit][ix.numpy()].astype("float32"))); l.backward(); opt.step()
        net.eval()
        with torch.no_grad(): pv=torch.sigmoid(net(xv)).numpy()
        th=choose_threshold(y[val],pv)
        fp=copy.deepcopy(net).cpu().eval()
        q=torch.ao.quantization.quantize_dynamic(copy.deepcopy(fp),{nn.Linear},dtype=torch.qint8)
        for name,mdl in [("FP32",fp),("INT8",q)]:
            p,med,p95=infer(mdl,xt)
            r=metrics(y[te],p,th); r.update(Precision=precision_score(y[te],p>=th,zero_division=0),
                Model=name,Seed=seed,MedianLatencyMs=med,P95LatencyMs=p95,SizeMB=size_mb(mdl))
            rows.append(r)
    raw=pd.DataFrame(rows); raw.to_csv(out/"fp32_int8_seed_metrics.csv",index=False)
    sm=summarize(raw,["Model"]); sm.to_csv(out/"fp32_int8_summary.csv",index=False)
    fp=sm[sm.Model=="FP32"].iloc[0]; qi=sm[sm.Model=="INT8"].iloc[0]
    extra=pd.DataFrame([{"SizeReductionPct":100*(fp.SizeMB_mean-qi.SizeMB_mean)/fp.SizeMB_mean,
                         "AUROCChange":qi.AUROC_mean-fp.AUROC_mean,
                         "AUPRCChange":qi.AUPRC_mean-fp.AUPRC_mean,
                         "BrierChange":qi.Brier_mean-fp.Brier_mean,
                         "FSRChange":qi.FSR_mean-fp.FSR_mean,
                         "FURChange":qi.FUR_mean-fp.FUR_mean,
                         "LatencyRatio":qi.MedianLatencyMs_mean/fp.MedianLatencyMs_mean}])
    extra.to_csv(out/"quantization_deltas.csv",index=False)
    (out/"fp32_int8_summary.tex").write_text(sm.to_latex(index=False,float_format="%.5f"),encoding="utf-8")
    print(sm.to_string(index=False)); print("\\nQuantization deltas:\\n",extra.to_string(index=False))
if __name__=="__main__": main()
