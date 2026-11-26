#!/usr/bin/env python
"""Reviewer experiment: practical driving-decision scenario validation."""

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

DABRA_GROUPS={
"behavior":["latitude","longitude","speed","acceleration","steering_angle","heading","trip_duration","trip_distance",
"fuel_consumption","rpm","brake_usage","lane_deviation","route_deviation_score","acceleration_variation","behavioral_consistency_index"],
"environment":["weather_conditions","road_type","traffic_condition"]}

def scenario_masks(df):
    # Observed scenario definitions only. No synthetic driving labels.
    masks={"All_test":np.ones(len(df),dtype=bool)}
    def add(name,col,vals):
        if col in df:
            s=df[col].astype(str).str.lower()
            m=s.isin([str(v).lower() for v in vals]).to_numpy()
            if m.sum()>=100: masks[name]=m
    add("Heavy_traffic","traffic_condition",["heavy","heavy traffic","high"])
    add("Highway","road_type",["highway"])
    add("Urban","road_type",["urban","city"])
    add("Rain","weather_conditions",["rain","rainy"])
    add("Fog","weather_conditions",["fog","foggy"])
    if "brake_usage" in df:
        x=pd.to_numeric(df.brake_usage,errors="coerce"); q=x.quantile(.75); masks["High_brake_usage"]=(x>=q).fillna(False).to_numpy()
    if "lane_deviation" in df:
        x=pd.to_numeric(df.lane_deviation,errors="coerce"); q=x.quantile(.75); masks["High_lane_deviation"]=(x>=q).fillna(False).to_numpy()
    if "speed" in df:
        x=pd.to_numeric(df.speed,errors="coerce"); q=x.quantile(.75); masks["High_speed"]=(x>=q).fillna(False).to_numpy()
    return masks

def main():
    out=OUTROOT/"decision_scenarios"; out.mkdir(parents=True,exist_ok=True)
    df,y=load_dabra(); cols=sum(DABRA_GROUPS.values(),[])
    rows=[]
    for seed in SEEDS:
        tr,te=split_idx(y,seed); fit,val=fit_val(tr,y,seed); model,cols2=pipe_for(df,cols)
        model.fit(df.iloc[fit][cols2],y[fit]); pv=model.predict_proba(df.iloc[val][cols2])[:,1]; th=choose_threshold(y[val],pv)
        pt=model.predict_proba(df.iloc[te][cols2])[:,1]
        local=df.iloc[te].reset_index(drop=True); yy=y[te]
        for name,mask in scenario_masks(local).items():
            if mask.sum()<50 or len(np.unique(yy[mask]))<2: continue
            r=metrics(yy[mask],pt[mask],th); r.update(Scenario=name,Seed=seed,N=int(mask.sum()))
            rows.append(r)
    raw=pd.DataFrame(rows); raw.to_csv(out/"decision_scenario_seed_metrics.csv",index=False)
    sm=summarize(raw,["Scenario"]); sm.to_csv(out/"decision_scenario_summary.csv",index=False)
    (out/"decision_scenario_summary.tex").write_text(sm.to_latex(index=False,float_format="%.4f"),encoding="utf-8")
    print(sm.to_string(index=False))
if __name__=="__main__": main()
