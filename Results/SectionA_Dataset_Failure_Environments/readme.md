# Section A — Dataset Failure Environments

This directory contains **all figures, tables, text summaries, and metadata files** used in **Section A (Dataset and Failure Environment Analysis)** of the *Causal-Invariant Digital Twin (CIDT)* study.  
The contents provide transparent evidence of dataset composition, environment-wise imbalance, and failure distributions used to motivate causal-invariant validation.

---

## 📊 Figures (Visual Artifacts)

### 1. `env_full_balance.png`
**Type:** Figure (PNG)  
**Description:**  
Visualizes the full dataset balance across all environment categories (e.g., weather, road type, traffic conditions).

**Used in Paper:**  
- Section A: Dataset Characterization  
- Demonstrates environment imbalance and motivates robustness analysis

---

## 📄 Text Files (Statistical Summaries)

### 2. `SectionA_Dataset_Failure_Environments-failure_distribution.txt`
**Type:** Text file  
**Description:**  
Contains environment-wise failure counts and proportions extracted from experimental runs.

**Used in Paper:**  
- Section A: Failure Environment Analysis  
- Supports claims on failure skew under different operating conditions

---

## 🗂️ Structured Data / Tables (Machine-Readable)

### 3. `environment_cardinality.json`
**Type:** JSON (Table-equivalent metadata)  
**Description:**  
Provides structured cardinality information for each environment category, including:
- Environment identifiers
- Sample counts per environment

**Used in Paper:**  
- Reproducibility and dataset transparency
- Programmatic loading for analysis scripts

**Role:**  
Serves as the authoritative source for all environment distribution tables reported in Section A.

---

## 📝 Documentation

### 4. `readme.md`
**Type:** Markdown documentation  
**Description:**  
Explains the purpose and contents of this directory.

---

## 📌 Summary of Included Artifacts

| Category | Files |
|--------|------|
| Figures | `env_full_balance.png` |
| Text Files | `SectionA_Dataset_Failure_Environments-failure_distribution.txt` |
| Tables / Metadata | `environment_cardinality.json` |
| Documentation | `readme.md` |

---

## 🔍 Role in CIDT Evaluation Pipeline

These artifacts collectively support:
- Dataset transparency (IEEE reproducibility requirements)
- Explicit modeling of environment-induced distribution shifts
- Justification for causal-invariant and interventional evaluation in CIDT

---

## 📖 Related Paper Section
- **Section:** Section A — Dataset and Failure Environment Analysis  
- **Project:** Causal-Invariant Digital Twin for Robust Autonomous Decision Validation under 5G-V2X Distribution Shifts

---

For further experimental details, refer to:
- `Results/SectionB_Baseline_Models_*`
- Root project `README.md`
