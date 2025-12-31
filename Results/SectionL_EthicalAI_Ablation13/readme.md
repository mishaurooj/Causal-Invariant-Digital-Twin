# Section L — Ethical AI & Fairness Evaluation (Ablation 13)

This directory contains **ethical AI and fairness evaluation results**
for all baseline, robustness-oriented, and causal models evaluated
in the *Causal-Invariant Digital Twin (CIDT)* study.

Due to GitHub rendering limitations, numerical results stored in
`.xlsx`, `.csv`, and `.txt` files are **previewed below as colored tables**.
The original files remain the **authoritative source** for analysis and reproduction.

---

## 📊 Fairness Comparison Across Models (Preview)

### Group-wise Fairness Metrics (CIDT vs Baselines)
**Source file:** `SectionL_EthicalAI_AllModels.xlsx`

<table>
<tr style="background-color:#0f172a;color:white;">
<th>Model</th>
<th>AUROC (Mean)</th>
<th>Risk Stability</th>
<th>False-Safe Rate</th>
<th>Fairness Behavior</th>
</tr>

<tr style="background-color:#ecfeff;">
<td><b>CIDT (100%)</b></td>
<td><b>0.86</b></td>
<td style="color:green;"><b>Stable</b></td>
<td style="color:green;"><b>0.00</b></td>
<td style="color:green;"><b>Invariant</b></td>
</tr>

<tr style="background-color:#f0fdf4;">
<td>CIDT (50%)</td>
<td>0.84</td>
<td style="color:green;">Stable</td>
<td style="color:green;">0.00</td>
<td>Invariant</td>
</tr>

<tr style="background-color:#fefce8;">
<td>IRM</td>
<td>0.71</td>
<td>Partially Stable</td>
<td style="color:red;">1.00</td>
<td>Collapsed</td>
</tr>

<tr style="background-color:#fff7ed;">
<td>DANN</td>
<td>0.69</td>
<td>Moderate</td>
<td style="color:red;">1.00</td>
<td>Majority Bias</td>
</tr>

<tr style="background-color:#fee2e2;">
<td>GroupDRO</td>
<td>0.66</td>
<td>Unstable</td>
<td style="color:red;">1.00</td>
<td>Worst-Group Overfit</td>
</tr>

<tr style="background-color:#f8fafc;">
<td>ERM (LogReg / MLP / XGB)</td>
<td>≈ 0.50</td>
<td>Degenerate</td>
<td style="color:red;">1.00</td>
<td>Trivial Predictions</td>
</tr>
</table>

➡ Full per-group metrics available in:  
📄 `SectionL_EthicalAI_AllModels.xlsx`

---

## 📄 CIDT Fairness vs Sample Size (Preview)

### CIDT Group Fairness Across Data Fractions
**Source files:**
- `fairness_CIDT_10pct.csv`
- `fairness_CIDT_25pct.csv`
- `fairness_CIDT_50pct.csv`
- `fairness_CIDT_100pct.csv`

<table>
<tr style="background-color:#020617;color:white;">
<th>Training Fraction</th>
<th>AUROC</th>
<th>Mean Risk</th>
<th>False-Safe Rate</th>
</tr>

<tr style="background-color:#ecfeff;">
<td>10%</td>
<td>0.81</td>
<td>0.32</td>
<td style="color:green;"><b>0.00</b></td>
</tr>

<tr style="background-color:#ecfeff;">
<td>25%</td>
<td>0.83</td>
<td>0.31</td>
<td style="color:green;"><b>0.00</b></td>
</tr>

<tr style="background-color:#ecfeff;">
<td>50%</td>
<td>0.84</td>
<td>0.30</td>
<td style="color:green;"><b>0.00</b></td>
</tr>

<tr style="background-color:#ecfeff;">
<td><b>100%</b></td>
<td><b>0.86</b></td>
<td><b>0.29</b></td>
<td style="color:green;"><b>0.00</b></td>
</tr>
</table>

---

## 📄 Baseline Fairness Result Files

The following CSV files contain **group-wise fairness metrics**
(AUROC, mean risk, false-safe rate) used in the paper:

- `fairness_LogReg_FP32.csv`
- `fairness_MLP_FP32.csv`
- `fairness_XGBoost_FP32.csv`
- `fairness_IRM_Full.csv`
- `fairness_DANN.csv`
- `fairness_GroupDRO.csv`
- `fairness_SEFN_FP32.csv`
- `fairness_SEFN_Temporal_FP32.csv`
- `fairness_SEFN_INT8.csv`

---

## 📄 Aggregate Logs

**File:** `allresults.txt`  
Contains consolidated fairness summaries and sanity checks across all models.

---

## 📌 Artifact Summary

| Category | Files |
|--------|------|
| Excel Tables | `SectionL_EthicalAI_AllModels.xlsx` |
| CSV Fairness | CIDT, ERM, IRM, DANN, GroupDRO, SEFN |
| Logs | `allresults.txt` |
| Documentation | `readme.md` |

---

## 🔍 Role in CIDT Framework

This ablation demonstrates that:
- CIDT maintains **fairness invariance across sensitive groups**
- Baselines collapse to trivial or biased predictions
- Robustness alone does not guarantee ethical behavior
- **Causal invariance is necessary for ethical AI guarantees**

These results establish CIDT as both **robust and ethically stable**.

---

## 📖 Related Paper Section
**Section:** Section L — Ethical AI and Fairness Evaluation  
**Project:** Causal-Invariant Digital Twin for Robust Autonomous Decision Validation

