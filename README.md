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
<span style="color:green;font-weight:bold;">✔</span> Explicitly supported &nbsp;&nbsp;
<span style="color:red;font-weight:bold;">✘</span> Not supported &nbsp;&nbsp;
<span style="color:#f59e0b;font-weight:bold;">◐</span> Partial / indirect

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

**Key takeaway:**  
> Existing DT frameworks address *components* of robustness, but **only CIDT jointly supports causal learning, interventions, ethical stability, and edge feasibility for decision validation under distribution shifts**.

---
---

## 📌 Paper Contributions

This repository accompanies the paper:

**“Causal-Invariant Digital Twin for Robust Autonomous Vehicle Decision Validation under 5G-V2X Distribution Shifts”**

The main scientific contributions are summarized as follows:

1. **Causal-Invariant Digital Twin (CIDT)**  
   A novel digital twin framework that validates *autonomous vehicle decisions* under distribution shifts by enforcing **causal invariance**, rather than optimizing predictive accuracy or control performance.

2. **Sensor–Event Fault Network (SEFN)**  
   A dedicated reliability modeling module that estimates **latent sensing, event, and V2X communication faults** without leaking fault information into prediction, enabling principled causal attribution and conservative risk gating.

3. **Invariant Representation Learning for Decision Validation**  
   The framework integrates **Invariant Risk Minimization (IRM)** to suppress environment- and communication-dependent spurious correlations, ensuring that validated decisions rely on stable causal mechanisms.

4. **Structural Causal Digital Twin with Interventions**  
   A **structural causal model (SCM)** is constructed over latent vehicle, driver, environment, communication, fault, action, and outcome variables, enabling **do-calculus–based interventional evaluation** without retraining.

5. **Intervention-Based Robustness Metrics**  
   CIDT introduces and evaluates robustness using **failure probability, invariant risk gap, intervention consistency score, and intervention sensitivity index**, moving beyond accuracy-centric validation.

6. **Ethical AI Validation under Distribution Shift**  
   Ethical reliability is assessed via **group-wise risk stability and false-safe analysis**, demonstrating that causal invariance yields fair and consistent behavior without explicit demographic conditioning.

7. **Edge-Feasible Deployment Analysis**  
   Post-training dynamic quantization is evaluated to study **efficiency–validity trade-offs**, showing that CIDT preserves validation integrity under realistic edge constraints.

---

## 📊 Dataset: Driver Behavior and Route Anomaly Detection (DBRA24)

This study uses the **DBRA24** real-world autonomous driving dataset, publicly available on Kaggle:

🔗 **Dataset Link:**  
https://www.kaggle.com/datasets/datasetengineer/driver-behavior-and-route-anomaly-dataset-dbra24

**Citation (BibTeX):**
```bibtex
@misc{dbra24kaggle,
  author       = {Dataset Engineer},
  title        = {{Driver Behavior and Route Anomaly Detection (DBRA24)}},
  howpublished = {\url{https://www.kaggle.com/datasets/datasetengineer/driver-behavior-and-route-anomaly-dataset-dbra24}},
  year         = {2024},
  note         = {Accessed: 2025-10-08}
}

---

## 🧠 CIDT Architecture

<p align="center">
  <img src="figs/structural_causal_twin_model.png" width="90%">
</p>

### Key Components
1. **Sensor Event Fault Network (SEFN)**  
2. **Invariant Representation Learning (IRM)**  
3. **Structural Causal Digital Twin (SCM-DT)**  
4. **Interventional Robustness Evaluation**

---

## 🔗 Learned Causal Structure

<p align="center">
  <img src="figs/causal_graph.png" width="65%">
</p>

The CIDT explicitly models causal relations between:
- vehicle dynamics,
- driver intent,
- environment context,
- V2X latency and reliability.

---

## 🔁 Interventional Validation (What Others Cannot Do)

<p align="center">
  <img src="Results/SectionF_CausalGraph_Interventions/interventional_analysis.png" width="85%">
</p>

CIDT performs **do-interventions** such as:
- `do(V2X_latency = high)`
- `do(driver_behavior = aggressive)`
- `do(weather = adverse)`

and **quantifies decision failure probability under counterfactual shifts**.

---

## 📊 Experimental Coverage (Sections A–L)

| Section | Focus |
|-------|------|
| A | Dataset & Failure Environments |
| B | ERM Baselines |
| C | Invariant Risk Minimization |
| D | Domain-Adversarial Learning (DANN) |
| E | Worst-Group Risk (GroupDRO) |
| F | Causal Graph & Interventions |
| G | Driver Held-Out & Spurious Ablations |
| J | Sample Efficiency |
| K | Event Sensor Fault Modeling (SEFN) |
| L | Ethical AI & Fairness |

---

## 📉 Why Correlation-Based Robustness Fails

<p align="center">
  <img src="figs/erm-irm.png" width="80%">
</p>

<p align="center">
  <img src="figs/dann.png" width="70%">
  <img src="figs/groupdro.png" width="70%">
</p>

| Method | Robust Under Shift? | Failure Mode |
|------|-------------------|-------------|
| ERM | ❌ | Trivial predictions |
| IRM | ❌ | False-safe collapse |
| DANN | ❌ | Majority bias |
| GroupDRO | ❌ | Worst-group overfit |
| **CIDT** | ✅ | Causal invariance |

---

## ⚠️ Event-Level Fault Modeling (SEFN)

<p align="center">
  <img src="figs/sefn.png" width="70%">
</p>

<p align="center">
  <img src="figs/sefn_ablation_radar_plus_event_importance.png" width="80%">
</p>

SEFN models **sensor and event failures explicitly** before causal inference.

---

## 📈 Sample Efficiency (Ablation 11)

<p align="center">
  <img src="figs/sample_efficiency_auroc_convergence.png" width="75%">
</p>

CIDT maintains strong performance with **only 10–25% of training data**.

---

## ⚖️ Ethical AI & Fairness (Ablation 13)

<p align="center">
  <img src="figs/fairness_NxN_all_metrics.png" width="85%">
</p>

### Key Observation
- Robustness ≠ Fairness
- **Only causal invariance yields ethical stability**

---

## 📊 Quantitative Summary (Reviewer Snapshot)

<table>
<tr style="background-color:#0f172a;color:white;">
<th>Model</th>
<th>OOD AUROC</th>
<th>False-Safe Rate</th>
<th>Fairness Stable</th>
<th>Interventions</th>
</tr>

<tr style="background-color:#ecfeff;">
<td><b>CIDT</b></td>
<td><b>0.86</b></td>
<td style="color:green;"><b>0.00</b></td>
<td style="color:green;"><b>Yes</b></td>
<td style="color:green;"><b>Yes</b></td>
</tr>

<tr style="background-color:#fff7ed;">
<td>IRM</td>
<td>0.71</td>
<td style="color:red;">1.00</td>
<td>No</td>
<td>No</td>
</tr>

<tr style="background-color:#fee2e2;">
<td>DANN</td>
<td>0.69</td>
<td style="color:red;">1.00</td>
<td>No</td>
<td>No</td>
</tr>

<tr style="background-color:#fef2f2;">
<td>GroupDRO</td>
<td>0.66</td>
<td style="color:red;">1.00</td>
<td>No</td>
<td>No</td>
</tr>
</table>

---

## 📂 Repository Structure


Each `Results/SectionX_*` directory contains a **self-rendering README** with figures, tables, and logs.

---

## 🧾 Reproducibility

- All metrics reported in CSV/XLSX/TXT form
- Colored table previews included for transparency
- Deterministic seeds and logged runs
- Apache 2.0 licensed

---

## 🏁 Key Takeaway (Editor-Level)

> **Robustness without causality is illusionary.**  
> CIDT is the **first Digital Twin framework** that validates autonomous decisions  
> **under distribution shifts using causal invariance and interventions**.

---

## 📜 License
Apache License 2.0

---

## 📬 Contact
For questions, clarifications, or reproducibility inquiries, please open an issue.
