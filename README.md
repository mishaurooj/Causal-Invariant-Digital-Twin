# Causal-Invariant Digital Twin for Robust Autonomous Vehicle Decision Validation under V2X Distribution Shifts

This repository contains the code, datasets, experimental outputs, and
revision experiments associated with the paper **"Causal-Invariant
Digital Twin for Robust Autonomous Vehicle Decision Validation under V2X
Distribution Shifts."**

## Repository

Main repository:

https://github.com/mishaurooj/Causal-Invariant-Digital-Twin/tree/main

The main repository is organized into `Code`, `Dataset`, `Results`, and
`figs`. The current reviewer-revision scripts are located in
`Code/Rev2`.

## Environment Setup

Python 3.10 or later is recommended.

Clone the repository and enter the project directory:

``` bash
git clone https://github.com/mishaurooj/Causal-Invariant-Digital-Twin.git
cd Causal-Invariant-Digital-Twin
```

Create and activate a virtual environment.

### Windows

``` bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade `pip` and install the packages used by the experiment scripts:

``` bash
python -m pip install --upgrade pip
pip install numpy pandas scipy scikit-learn matplotlib torch xgboost joblib pyarrow
```

## Dataset Setup

The experiments use **DABRA** and **VeReMi** as independent dataset
instantiations. They must not be row-joined, timestamp-aligned, or
treated as paired observations.

Create or use the repository `Dataset` directory:

``` text
Causal-Invariant-Digital-Twin/
└── Dataset/
    ├── DABRA.csv
    └── VeReMi.csv
```

The repository currently includes the archived DABRA data file:

``` text
Dataset/driver_behavior_route_anomaly_dataset_with_derived_features.rar
```

Extract the archive and prepare the DABRA CSV expected by the revision
scripts as:

``` text
Dataset/DABRA.csv
```

The revision scripts expect the VeReMi data as:

``` text
Dataset/VeReMi.csv
```

`VeReMi.csv` is not currently present in the repository, so it must be
prepared separately before running experiments that require VeReMi.

Some revision scripts currently contain a local Windows project path
such as:

``` python
ROOT = Path(r"E:\other\Causal-DT")
```

Before running those scripts on another machine, change `ROOT` or
`PROJECT_ROOT` to the local repository path.

## Main Experiment

The main CIDT reviewer experiment is:

``` text
Code/Rev2/run_cidt_reviewers.py
```

View its command-line options before running:

``` bash
python Code/Rev2/run_cidt_reviewers.py --help
```

The script contains the main CIDT experimental pipeline, including
predictive evaluation, calibration, distribution-shift evaluation,
invariant learning, and associated result generation.

## Additional Revision Experiments

The additional reviewer experiments are located in `Code/Rev2`:

``` text
reviewer_component_ablation.py
reviewer_decision_scenario_validation.py
reviewer_sctm_intervention_validation.py
reviewer_int8_deployment_validation.py
```

Run them from the repository root after setting the dataset and project
paths:

``` bash
python Code/Rev2/reviewer_component_ablation.py
python Code/Rev2/reviewer_decision_scenario_validation.py
python Code/Rev2/reviewer_sctm_intervention_validation.py
python Code/Rev2/reviewer_int8_deployment_validation.py
```

These scripts respectively evaluate component-level ablation, DABRA
driving-decision scenarios, SCTM controlled interventions, and
deployment-stage FP32 versus dynamic INT8 quantization.

## Experimental Protocol

The revised experiments use five random seeds:

``` text
5, 11, 23, 37, 45
```

DABRA and VeReMi are evaluated independently. DABRA supports the
driving-risk and decision-validation experiments, while VeReMi supports
the V2X fault/misbehavior experiments. Results from the two datasets
should not be interpreted as observation-level data fusion.

## Outputs

Experiment outputs are written by the individual scripts to their
configured result directories. The repository also contains a top-level
`Results` directory for experimental results and a `figs` directory for
figures.

## Reproducibility Notes

Before running an experiment:

1.  Confirm that `DABRA.csv` and, when required, `VeReMi.csv` are
    available in the configured dataset directory.
2.  Update any hard-coded `ROOT` or `PROJECT_ROOT` path in the revision
    scripts.
3.  Run the scripts from the repository root.
4.  Keep DABRA and VeReMi as independent experimental datasets.
5.  Use the five specified seeds when reproducing the revised multi-run
    results.

## License

Apache License 2.0.
