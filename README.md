# Causal-Invariant Digital Twin (CIDT)

### Robust Autonomous Vehicle Decision Validation under 5G-V2X Distribution Shifts  
**IEEE Transactions–Ready Research Repository**

---

<p align="center">
  <img src="figs/cidt.png" width="85%">
</p>

---

## 🔍 Overview

Autonomous Vehicle (AV) decision-making systems increasingly rely on **V2X communication, heterogeneous sensing, and learned prediction models**.  
However, **most Digital Twin (DT) frameworks remain correlation-driven**, leading to **catastrophic failure under distribution shifts** such as:

- unseen traffic patterns,
- new driver behaviors,
- degraded V2X reliability,
- weather and sensing perturbations.

This repository presents the **first end-to-end Causal-Invariant Digital Twin (CIDT)** framework that **explicitly validates AV decisions under distribution shifts using causal invariance and intervention-based simulation**.

---

## 🚨 Core Research Gap (Why This Work Exists)

| Limitation in Existing DTs | Consequence |
|---------------------------|-------------|
| Correlation-based validation | Unsafe decision transfer |
| No causal disentanglement | Spurious robustness |
| No interventions | No counterfactual guarantees |
| Predictive-only evaluation | False sense of safety |

**CIDT directly addresses all four gaps.**

---

## 📚 Literature Comparison: Digital Twin Decision Validation

**Legend:**  
✔ Explicitly supported &nbsp;&nbsp; ✘ Not supported &nbsp;&nbsp; ◐ Partial / indirect

<table>
<thead>
<tr style="background-color:#0f172a;color:white;">
<th>Work</th>
<th>Primary Technique</th>
<th>Eval. Focus</th>
<th>Key Limitation</th>
<th>DT</th>
<th>Learn</th>
<th>V2X</th>
<th>Causal</th>
<th>Interv.</th>
<th>Edge</th>
<th>Ethics</th>
</tr>
</thead>
<tbody>

<tr style="background-color:#f8fafc;">
<td>Wu et al. (2025)</td>
<td>V2X corridor DT sync</td>
<td>Integration accuracy</td>
<td>No decision robustness</td>
<td style="color:green;">✔</td>
<td style="color:red;">✘</td>
<td style="color:green;">✔</td>
<td style="color:red;">✘</td>
<td style="color:red;">✘</td>
<td style="color:red;">✘</td>
<td style="color:red;">✘</td>
</tr>

<tr style="background-color:#ecfeff;">
<td>Li et al. (2025)</td>
<td>Risk-aware mobility DT</td>
<td>Situational awareness</td>
<td>Correlation-based</td>
<td style="color:green;">✔</td>
<td style="color:green;">✔</td>
<td style="color:green;">✔</td>
<td style="color:red;">✘</td>
<td style="color:red;">✘</td>
<td style="color:#f59e0b;">◐</td>
<td style="color:red;">✘</td>
</tr>

<tr style="background-color:#fefce8;">
<td>Xun et al. (2025)</td>
<td>DT trajectory prediction</td>
<td>Prediction accuracy</td>
<td>Shift sensitive</td>
<td style="color:green;">✔</td>
<td style="color:green;">✔</td>
<td style="color:red;">✘</td>
<td style="color:red;">✘</td>
<td style="color:red;">✘</td>
<td style="color:red;">✘</td>
<td style="color:red;">✘</td>
</tr>

<tr style="background-color:#fff7ed;">
<td>Lim et al. (2024)</td>
<td>Graph-based causal DT</td>
<td>Root-cause inference</td>
<td>Not AV decisions</td>
<td style="color:green;">✔</td>
<td style="color:green;">✔</td>
<td style="color:red;">✘</td>
<td style="color:green;">✔</td>
<td style="color:red;">✘</td>
<td style="color:red;">✘</td>
<td style="color:red;">✘</td>
</tr>

<tr style="background-color:#fee2e2;">
<td>Thomas et al. (2023)</td>
<td>Causal imitation learning</td>
<td>Generalization</td>
<td>No safety validation</td>
<td style="color:green;">✔</td>
<td style="color:green;">✔</td>
<td style="color:green;">✔</td>
<td style="color:green;">✔</td>
<td style="color:red;">✘</td>
<td style="color:red;">✘</td>
<td style="color:red;">✘</td>
</tr>

<tr style="background-color:#e0f2fe;">
<td><b>SEFN (2026)</b></td>
<td>Event-driven fault modeling</td>
<td>Fault reliability</td>
<td>—</td>
<td style="color:red;">✘</td>
<td style="color:green;">✔</td>
<td style="color:green;">✔</td>
<td style="color:red;">✘</td>
<td style="color:red;">✘</td>
<td style="color:green;">✔</td>
<td style="color:green;">✔</td>
</tr>

<tr style="background-color:#dbeafe;">
<td><b>SCTM (2026)</b></td>
<td>Structural causal DT</td>
<td>Causal attribution</td>
<td>No invariance</td>
<td style="color:green;">✔</td>
<td style="color:red;">✘</td>
<td style="color:green;">✔</td>
<td style="color:green;">✔</td>
<td style="color:green;">✔</td>
<td style="color:green;">✔</td>
<td style="color:green;">✔</td>
</tr>

<tr style="background-color:#bbf7d0;">
<td><b>CIDT (2026)</b></td>
<td><b>Causal-invariant DT</b></td>
<td><b>Robust decision validation</b></td>
<td><b>—</b></td>
<td style="color:green;font-weight:bold;">✔</td>
<td style="color:green;font-weight:bold;">✔</td>
<td style="color:green;font-weight:bold;">✔</td>
<td style="color:green;font-weight:bold;">✔</td>
<td style="color:green;font-weight:bold;">✔</td>
<td style="color:green;font-weight:bold;">✔</td>
<td style="color:green;font-weight:bold;">✔</td>
</tr>

</tbody>
</table>

---

## 📌 Paper Contributions

This repository accompanies the paper:

**“Causal-Invariant Digital Twin for Robust Autonomous Vehicle Decision Validation under 5G-V2X Distribution Shifts”**

Key contributions include causal-invariant validation, sensor–event fault modeling, structural causal twins with interventions, ethical AI robustness, and edge-feasible deployment.

---

## 📊 Dataset: Driver Behavior and Route Anomaly Detection (DBRA24)

**Dataset Link:**  
https://www.kaggle.com/datasets/datasetengineer/driver-behavior-and-route-anomaly-dataset-dbra24

**BibTeX citation:**
```bibtex
@misc{dbra24kaggle,
  author       = {Dataset Engineer},
  title        = {{Driver Behavior and Route Anomaly Detection (DBRA24)}},
  howpublished = {\url{https://www.kaggle.com/datasets/datasetengineer/driver-behavior-and-route-anomaly-dataset-dbra24}},
  year         = {2024},
  note         = {Accessed: 2025-10-08}
}
```

---

---

## 🖼️ Experimental Evidence
---

### Figure 1: Overall CIDT Framework
<p align="center">
  <img src="figs/cidt.png" width="100%">
</p>

**Explanation:**  
This figure illustrates the complete **Causal-Invariant Digital Twin (CIDT)** pipeline, including sensor–event fault modeling, invariant representation learning, structural causal twin construction, and interventional decision validation. The sequential design ensures that uncertainty from sensing and communication is handled *prior* to causal reasoning.

---

### Figure 2: Structural Causal Twin Model
<p align="center">
  <img src="figs/structural_causal_twin_model.png" width="100%">
</p>

**Explanation:**  
The structural causal model (SCM) defines causal dependencies among latent vehicle dynamics, driver behavior, environment context, communication reliability, fault variables, actions, and outcomes. This representation enables counterfactual reasoning via do-interventions.

---

### Figure 3: Learned Causal Graph
<p align="center">
  <img src="figs/causal_graph.png" width="1005%">
</p>

**Explanation:**  
The learned causal graph captures directional causal relations inferred from invariant latent representations. Edges represent structural dependencies used during interventional simulation, not correlations learned directly from raw features.

---

### Figure 4: Causal Effect Decomposition
<p align="center">
  <img src="figs/causal_effect_decomposition..png" width="100%">
</p>

**Explanation:**  
This figure decomposes the total decision outcome into direct, indirect, and mediated causal effects. It highlights how environmental and communication factors influence decisions through specific causal pathways.

---

### Figure 5: Environment Distribution by Name
<p align="center">
  <img src="figs/env_distribution_by_name_WITH_LEGEND.png" width="100%">
</p>

**Explanation:**  
The distribution of samples across environments defined by weather, road type, and traffic density. These environment partitions are used to induce controlled distribution shifts for robustness evaluation.

---

### Figure 6: ERM vs IRM under Distribution Shift
<p align="center">
  <img src="figs/erm-irm.png" width="80%">
</p>

**Explanation:**  
Comparison of empirical risk minimization (ERM) and invariant risk minimization (IRM). While IRM enforces invariance, both approaches fail to provide reliable decision validation under strong shifts, motivating causal modeling.

---

### Figure 7: Domain-Adversarial Neural Network (DANN)
<p align="center">
  <img src="figs/dann.png" width="70%">
</p>

**Explanation:**  
DANN reduces domain distinguishability but collapses to majority-class predictions under shift, resulting in high false-safe rates despite apparent domain alignment.

---

### Figure 8: GroupDRO Worst-Group Optimization
<p align="center">
  <img src="figs/groupdro.png" width="70%">
</p>

**Explanation:**  
GroupDRO optimizes worst-case group risk but remains correlation-driven, overfitting minority groups and failing to generalize under unseen shifts.

---

### Figure 9: Predictive vs Invariant Metrics
<p align="center">
  <img src="figs/predictive_vs_invariant_all_metrics.png" width="85%">
</p>

**Explanation:**  
Predictive performance metrics diverge from invariant robustness metrics, demonstrating that high accuracy does not imply safe or stable decision validation.

---

### Figure 10: Loss Decomposition Analysis
<p align="center">
  <img src="figs/loss_decomposition.png" width="80%">
</p>

**Explanation:**  
Loss components are decomposed into predictive error, invariant penalty, and robustness gap, highlighting the contribution of invariance enforcement in CIDT.

---

### Figure 11: Fault Classification Comparison
<p align="center">
  <img src="figs/fault_classification_comparison.png" width="80%">
</p>

**Explanation:**  
Comparison of fault detection accuracy across models. SEFN achieves near-perfect fault discrimination without leaking fault information into decision prediction.

---

### Figure 12: Sensor–Event Fault Network (SEFN)
<p align="center">
  <img src="figs/sefn.png" width="70%">
</p>

**Explanation:**  
SEFN models latent reliability of sensors and events over time, producing fault likelihood estimates that are used exclusively for causal attribution and risk gating.

---

### Figure 13: SEFN Event Importance (Radar + Events)
<p align="center">
  <img src="figs/sefn_ablation_radar_plus_event_importance.png" width="85%">
</p>

**Explanation:**  
Event importance visualization showing how different sensing modalities and events contribute to fault likelihood estimation.

---

### Figure 14: Sample Efficiency AUROC Convergence
<p align="center">
  <img src="figs/sample_efficiency_auroc_convergence.png" width="80%">
</p>

**Explanation:**  
CIDT maintains stable validation performance with limited training data, demonstrating strong sample efficiency compared to correlation-based baselines.

---

### Figure 15: Fairness Metrics Across Groups
<p align="center">
  <img src="figs/fairness_NxN_all_metrics.png" width="85%">
</p>

**Explanation:**  
Group-wise evaluation of risk stability and false-safe rates across sensitive attributes. CIDT shows uniform behavior without explicit demographic conditioning.

---

### Figure 16: Grouped Experimental Results
<p align="center">
  <img src="figs/sectionG_grouped_bar_results.png" width="80%">
</p>

**Explanation:**  
Aggregated comparison across experimental sections, summarizing robustness, fairness, and stability trends across all evaluated methods.

---

### Summary Observation

Across all figures, results consistently show that **correlation-based robustness methods fail under distribution shift**, while **CIDT achieves stable, interpretable, and ethically consistent decision validation through causal invariance and interventions**.

---


---

## 📂 Repository Structure

```
Code/        → Training and evaluation scripts
Results/     → Section-wise experimental outputs (A–L)
figs/        → Publication-quality figures
README.md    → This document
```

---

## 🧾 Reproducibility

- Metrics provided in CSV / XLSX / TXT
- Deterministic seeds and logged runs
- Apache 2.0 license

---

## 🏁 Key Takeaway

**Robustness without causality is illusory.**  
CIDT validates autonomous decisions under distribution shifts using causal invariance and interventions.

---

## 📜 License
Apache License 2.0

---

## 📬 Contact
For questions or reproducibility inquiries, please open a GitHub issue.
