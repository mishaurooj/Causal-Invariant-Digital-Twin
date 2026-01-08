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
Key contributions include causal-invariant validation, sensor–event fault modeling, structural causal twins with interventions, ethical AI robustness, and edge-feasible deployment.

<div style="background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); padding:18px; border-radius:10px; color:white;">

The key contributions of this work are summarized below.
Each contribution is methodologically distinct and addresses a specific limitation of existing Digital Twin frameworks.

</div>
🧠 Causal-Invariant Digital Twin (CIDT)
<div style="background:#e3f2fd; padding:12px; border-left:6px solid #1565c0; border-radius:6px;"> A novel <b>Causal-Invariant Digital Twin (CIDT)</b> framework is introduced to validate <b>autonomous vehicle decision robustness</b> under <b>V2X-induced distribution shifts</b>, explicitly prioritizing <b>decision validation</b> over policy learning or predictive accuracy. </div>
🔒 Invariant Representation Learning
<div style="background:#e8f5e9; padding:12px; border-left:6px solid #2e7d32; border-radius:6px;"> An <b>invariant representation learning</b> module based on <b>Invariant Risk Minimization (IRM)</b> is employed to isolate <b>causal, decision-relevant factors</b> while suppressing <b>environment- and communication-dependent spurious correlations</b>. </div>
🔗 Structural Causal Twin Model (SCTM)
<div style="background:#fff3e0; padding:12px; border-left:6px solid #ef6c00; border-radius:6px;"> A <b>Structural Causal Twin Model (SCTM)</b> is formulated to enable <b>causal attribution</b> and <b>counterfactual reasoning</b>. Its behavior and inherent limitations under increasing distribution shift are systematically characterized. </div>
🧪 Intervention-Based Decision Validation
<div style="background:#f3e5f5; padding:12px; border-left:6px solid #6a1b9a; border-radius:6px;"> An <b>intervention-based validation mechanism</b> grounded in <b>do-calculus</b> is introduced to evaluate decision robustness under controlled perturbations in <b>sensing reliability, V2X communication quality, and driving behavior</b>. </div>
⚠️ Sensor–Event Fault Network (SEFN)
<div style="background:#fbe9e7; padding:12px; border-left:6px solid #c62828; border-radius:6px;"> A dedicated <b>Sensor–Event Fault Network (SEFN)</b> is developed to explicitly model <b>sensing, event, and communication unreliability</b>, enabling <b>fault-aware system state abstraction</b> without inflating predictive confidence. </div>
🚀 Edge-Feasible Deployment Analysis
<div style="background:#e0f2f1; padding:12px; border-left:6px solid #00695c; border-radius:6px;"> <b>Edge-feasible CIDT variants</b> are evaluated using <b>post-training dynamic quantization</b>, revealing efficiency–validity trade-offs and demonstrating that causal validation reliability can be preserved under deployment constraints. </div>
⚖️ Ethical Reliability Assessment
<div style="background:#ede7f6; padding:12px; border-left:6px solid #4527a0; border-radius:6px;"> An <b>ethical reliability evaluation</b> is conducted using <b>subgroup-wise risk stability</b> and <b>false-safe analysis</b>, ensuring consistent and trustworthy decision validation across heterogeneous operating conditions. </div>
✅ Summary Insight
<div style="background:#263238; color:white; padding:14px; border-radius:8px;"> Collectively, these contributions establish <b>causal invariance</b> as a necessary foundation for <b>trustworthy Digital Twin–based validation</b> of autonomous vehicle decisions under real-world distribution shifts. </div>

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

### Overall CIDT Framework
<p align="center">
  <img src="figs/cidt.png" width="100%">
</p>

**Explanation:**  
This figure illustrates the complete **Causal-Invariant Digital Twin (CIDT)** pipeline, including sensor–event fault modeling, invariant representation learning, structural causal twin construction, and interventional decision validation. The sequential design ensures that uncertainty from sensing and communication is handled *prior* to causal reasoning.

---

### Structural Causal Twin Model
<p align="center">
  <img src="figs/sctm.png" width="100%">
</p>

**Explanation:**  
The structural causal model (SCM) defines causal dependencies among latent vehicle dynamics, driver behavior, environment context, communication reliability, fault variables, actions, and outcomes. This representation enables counterfactual reasoning via do-interventions.

---

### Sensor--Event Fault Network (SEFN)
<p align="center">
  <img src="figs/sefn.png" width="100%">
</p>

**Explanation:**  
SEFN is a standalone reliability estimation model whose sole purpose is fault attribution. 

---

### Learned Causal Graph
<p align="center">
  <img src="figs/causal_graph.png" width="1005%">
</p>

**Explanation:**  
The learned causal graph captures directional causal relations inferred from invariant latent representations. Edges represent structural dependencies used during interventional simulation, not correlations learned directly from raw features.

---

### Causal Effect Decomposition
<p align="center">
  <img src="figs/causal_effect_decomposition..png" width="100%">
</p>

**Explanation:**  
This figure decomposes the total decision outcome into direct, indirect, and mediated causal effects. It highlights how environmental and communication factors influence decisions through specific causal pathways.

---

### DABRA-24 Dataset Environment Distribution 
<p align="center">
  <img src="figs/env_distribution_by_name_WITH_LEGEND.png" width="100%">
</p>

**Explanation:**  
The distribution of samples across environments defined by weather, road type, and traffic density. These environment partitions are used to induce controlled distribution shifts for robustness evaluation.

---

### Figure 9: Predictive vs Invariant Metrics
<p align="center">
  <img src="figs/predictive_vs_invariant_all_metrics.png" width="85%">
</p>

**Explanation:**  
Predictive performance metrics diverge from invariant robustness metrics, demonstrating that high accuracy does not imply safe or stable decision validation.

---

### Loss Decomposition Analysis
<p align="center">
  <img src="figs/loss_decomposition.png" width="80%">
</p>

**Explanation:**  
Loss components are decomposed into predictive error, invariant penalty, and robustness gap, highlighting the contribution of invariance enforcement in CIDT.


### Fault Classification Comparison
<p align="center">
  <img src="figs/fault_classification_comparison.png" width="80%">
</p>

**Explanation:**  
Comparison of fault detection accuracy across models. SEFN achieves near-perfect fault discrimination without leaking fault information into decision prediction.

---

### SEFN Event Importance (Radar + Events)
<p align="center">
  <img src="figs/sefn_ablation_radar_plus_event_importance.png" width="85%">
</p>

**Explanation:**  
Event importance visualization showing how different sensing modalities and events contribute to fault likelihood estimation.

---

### Sample Efficiency AUROC Convergence
<p align="center">
  <img src="figs/sample_efficiency_auroc_convergence.png" width="80%">
</p>

**Explanation:**  
CIDT maintains stable validation performance with limited training data, demonstrating strong sample efficiency compared to correlation-based baselines.

---

### Fairness Metrics Across Groups
<p align="center">
  <img src="figs/fairness_NxN_all_metrics.png" width="85%">
</p>

**Explanation:**  
Group-wise evaluation of risk stability and false-safe rates across sensitive attributes. CIDT shows uniform behavior without explicit demographic conditioning.

---

### Grouped Experimental Results
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

## 🏁 Key Takeaway

**Robustness without causality is illusory.**  
CIDT validates autonomous decisions under distribution shifts using causal invariance and interventions.

---

## 📜 License
Apache License 2.0

---
