#!/usr/bin/env python
"""Reviewer experiment: SCTM intervention-effectiveness and stability validation."""

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

# This is a controlled-risk validation because the observational datasets do not
# provide randomized intervention ground truth. It evaluates direction, magnitude,
# stability, bootstrap uncertainty, and predictive consistency of supported perturbations.

def bootstrap_effect(p0,p1,seed,B=500):
    rng=np.random.default_rng(seed); d=np.asarray(p1)-np.asarray(p0); n=len(d)
    vals=np.empty(B)
    for b in range(B):
        ix=rng.integers(0,n,n); vals[b]=d[ix].mean()
    return d.mean(),d.std(ddof=1),np.quantile(vals,.025),np.quantile(vals,.975)

def main():
    out=OUTROOT/"sctm_interventions"; out.mkdir(parents=True,exist_ok=True)
    df,y,groups=load_veremi(); cols=sum(groups.values(),[])
    rows=[]
    for seed in SEEDS:
        tr,te=split_idx(y,seed); fit,val=fit_val(tr,y,seed); model,cols2=pipe_for(df,cols)
        model.fit(df.iloc[fit][cols2],y[fit])
        p0=model.predict_proba(df.iloc[te][cols2])[:,1]
        base_auc=roc_auc_score(y[te],p0)
        for block,bcols in groups.items():
            use=[c for c in bcols if c in cols2]
            xcf=df.iloc[te][cols2].copy()
            # training-mean intervention in original feature scale
            for c in use:
                xcf[c]=pd.to_numeric(df.iloc[fit][c],errors="coerce").mean()
            p1=model.predict_proba(xcf)[:,1]
            mean,sd,lo,hi=bootstrap_effect(p0,p1,seed+len(block))
            delta=np.asarray(p1)-p0
            r=dict(Block=block,Seed=seed,N=len(te),
                BaselineRisk=float(p0.mean()),IntervenedRisk=float(p1.mean()),
                MeanRiskDifference=float(mean),AbsMeanRiskDifference=float(abs(mean)),
                EffectSD=float(sd),CI95Low=float(lo),CI95High=float(hi),
                MedianAbsChange=float(np.median(np.abs(delta))),
                MeanAbsChange=float(np.mean(np.abs(delta))),
                P95AbsChange=float(np.quantile(np.abs(delta),.95)),
                MaxAbsChange=float(np.max(np.abs(delta))),
                PositiveEffectRate=float((delta>0).mean()),
                NegativeEffectRate=float((delta<0).mean()),
                SignConsistency=float(max((delta>0).mean(),(delta<0).mean())),
                BaselineAUROC=float(base_auc),
                IntervenedAUROC=float(roc_auc_score(y[te],p1)),
                AUROCChange=float(roc_auc_score(y[te],p1)-base_auc),
                Correlation=float(np.corrcoef(p0,p1)[0,1]))
            rows.append(r)
    raw=pd.DataFrame(rows); raw.to_csv(out/"sctm_intervention_seed_metrics.csv",index=False)
    sm=summarize(raw,["Block"]); sm.to_csv(out/"sctm_intervention_summary.csv",index=False)
    (out/"sctm_intervention_summary.tex").write_text(sm.to_latex(index=False,float_format="%.6f"),encoding="utf-8")
    print("NOTE: observational controlled-risk sensitivity, not randomized causal ground truth.")
    print(sm.to_string(index=False))
if __name__=="__main__": main()
