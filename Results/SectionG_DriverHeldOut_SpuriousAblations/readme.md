# Section G — Driver Held-Out and Spurious Correlation Ablations

This directory contains **held-out driver evaluations and spurious correlation ablation results**
used in **Section G** of the *Causal-Invariant Digital Twin (CIDT)* study.

These experiments test whether models rely on **true causal factors or spurious shortcuts**.

---

## 📄 Held-Out Driver Evaluation

### Driver Held-Out Results
**File:** `driver_held_out_results.csv`

This file reports performance when:
- Entire driver identities are excluded during training
- Models are evaluated on unseen driver behavior

Metrics include AUROC, accuracy, and failure rates.

**Purpose:**  
Tests generalization beyond memorized driver-specific patterns.

---

## ⚠️ Spurious Correlation Ablation

### Spurious Feature Evaluation
**File:** `spurious_results.csv`

Contains results when:
- Spurious but predictive features are injected
- Causal signals are weakened or masked

Used to identify shortcut learning and correlation dependence.

---

## 📌 Artifact Summary

| Category | Files |
|--------|------|
| Held-Out Evaluation | `driver_held_out_results.csv` |
| Spurious Ablation | `spurious_results.csv` |
| Documentation | `readme.md` |

---

## 🔍 Role in CIDT Framework

This section demonstrates that:
- ERM, IRM, DANN, and GroupDRO degrade sharply under driver hold-out
- Spurious correlations inflate apparent robustness
- CIDT maintains stable behavior by relying on causal representations

These results **validate the necessity of causal invariance**.

---

## 📖 Related Paper Section
**Section:** Section G — Spurious Correlation and Driver Hold-Out Analysis  
**Project:** Causal-Invariant Digital Twin for Robust Autonomous Decision Validation

