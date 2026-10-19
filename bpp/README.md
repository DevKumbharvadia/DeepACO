# DeepACO — Optimized Bin Packing Using Adaptive Item Ordering

## Overview

DeepACO addresses the **Bin Packing Problem (BPP)**: packing items into fixed-capacity bins while minimizing the number of bins used.

The project combines:

- **Graph Neural Network (GNN):** Provides learned guidance for item placement.
- **Ant Colony Optimization (ACO):** Searches for good packing solutions using model-guided decisions.
- **Classical heuristics:** First Fit (FF), First Fit Decreasing (FFD), and Best Fit Decreasing (BFD) provide comparison baselines.
- **Local Search:** Attempts to improve DeepACO solutions through item moves, swaps, and bin elimination.

The project evaluates solution quality, unused capacity, runtime, solution validity, and scalability.

## Requirements

- Windows 10 or Windows 11
- Python 3.8
- Pretrained checkpoint: `../pretrained/bpp/bpp120.pt`
- Test dataset: `../data/bpp/testDataset-120.pt`

Run commands from the `bpp` project directory.

## Installation

Open PowerShell in the project directory.

If you are currently in the parent directory containing `bpp`, run:

```powershell
cd .\bpp
```

Create and activate the virtual environment:

```powershell
py -3.8 -m venv .venv-deepaco
.\.venv-deepaco\Scripts\Activate.ps1
```

Upgrade pip and install the dependencies:

```powershell
python -m pip install --upgrade "pip<25.1"
python -m pip install numpy==1.24.4
python -m pip install torch
python -m pip install scipy networkx matplotlib tqdm numba torch-geometric pandas
```

Verify the installation:

```powershell
python --version
python -c "import torch, torch_geometric, numpy, pandas; print('Dependencies OK')"
```

If PowerShell blocks environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-deepaco\Scripts\Activate.ps1
```

Then activate the environment again.

> **Compatibility note:** The commands above install the standard packages, but the appropriate PyTorch and PyTorch Geometric versions can depend on your Python version, operating system, and CPU/GPU configuration. If installation fails, check the compatible versions before changing the environment.

## How It Works

1. Load the pretrained GNN checkpoint and test dataset.
2. Construct bin-packing solutions using GNN-guided ACO.
3. Evaluate classical heuristics as baseline methods.
4. Apply Local Search to attempt to reduce the number of bins.
5. Validate item assignments and bin-capacity constraints.
6. Run scalability experiments across multiple item counts.
7. Analyze experiment results and generate CSV summaries and plots.

## Run the Project

### Recommended: Run the complete workflow

From the `bpp` directory, execute:

```powershell
python .\run_project.py
```

The master runner coordinates the following stages:

1. Classical baselines — `test_classical.py`
2. Original DeepACO baseline — `test.py`
3. DeepACO + Local Search — `test_deepaco_local_search.py`
4. Scalability experiments — `test_scalability.py`
5. Final results analysis — `analyze_results.py`

The default scalability configuration uses seven item counts:

`60, 90, 120, 150, 180, 210, 240`

With 20 instances per size, this corresponds to **140 scalability instances**.

The runner records each stage's status and runtime, continues to later stages if an individual test fails, and attempts final analysis after the experiment stages. A final summary reports passed, failed, skipped, and interrupted stages.

A full run can take considerable time. Experiment scripts may create new timestamped output files under `logs/`.

### Run scalability experiments only

```powershell
python .\run_project.py --scalability-only
```

Run all seven default sizes with 50 instances per size:

```powershell
python .\run_project.py --scalability-only --sizes 60 90 120 150 180 210 240 --instances 50
```

Run a smaller scalability test:

```powershell
python .\run_project.py --scalability-only --sizes 60 120 240 --instances 10
```

The `--seed` argument controls the seed passed to the scalability script:

```powershell
python .\run_project.py --scalability-only --seed 20261017
```

### Analyze existing results only

```powershell
python .\run_project.py --analysis-only
```

This runs `analyze_results.py` without rerunning the experiment stages. The analysis script must be able to locate the existing result files to produce meaningful reports.

### Skip selected stages

Skip classical baselines:

```powershell
python .\run_project.py --skip-classical
```

Skip the original DeepACO baseline:

```powershell
python .\run_project.py --skip-baseline
```

Skip Local Search:

```powershell
python .\run_project.py --skip-local-search
```

Skip scalability experiments:

```powershell
python .\run_project.py --skip-scalability
```

Multiple skip options can be combined.

### Run individual scripts

You can also execute stages independently:

```powershell
python .\test_classical.py
python .\test.py
python .\test_deepaco_local_search.py
python .\test_scalability.py
python .\analyze_results.py
```

Use the individual scripts when you want to run a specific experiment or debug a stage.

## Project Files

| File | Purpose |
|---|---|
| `run_project.py` | Coordinates experiment stages and produces a final execution summary |
| `net.py` | Graph Neural Network model |
| `aco.py` | Ant Colony Optimization implementation |
| `bfd.py` | Best Fit Decreasing heuristic |
| `local_search.py` | Local Search improvement methods |
| `utils.py` | Dataset, graph, and utility functions |
| `test.py` | Original DeepACO baseline evaluation |
| `test_classical.py` | Classical heuristic comparison |
| `test_deepaco_local_search.py` | DeepACO and Local Search comparison |
| `test_scalability.py` | Scalability experiments |
| `analyze_results.py` | Result analysis and visualization |
| `train.ipynb` | Notebook for training or experimentation, depending on its implementation |
| `logs/` | Experiment logs, CSV files, plots, and master-run logs |

Keep the existing file structure unchanged unless a deliberate project refactor is required.

## Results and Evaluation

The completed experiments evaluated five methods across seven item counts, with 20 instances per size, for a total of **140 scalability instances**.

### DeepACO vs. DeepACO + Local Search

| Metric | Result |
|---|---:|
| Improved instances | 33 |
| Unchanged instances | 107 |
| Worsened instances | 0 |
| Total bins saved | 33 |

These results reflect the completed experiment run and its tested instances. They do not guarantee the same performance on other datasets, random seeds, or problem sizes.

Classical heuristics were substantially faster in the completed experiments. Runtime and solution quality should therefore be considered together when comparing the methods.

### Analysis outputs

The final analysis directory is:

`logs/final_scalability_analysis/`

It contains the generated analysis artifacts, including available CSV summaries and plots for:

- Average bins used
- Average runtime
- Average unused capacity
- DeepACO versus Local Search
- Per-instance Local Search improvements

The exact files produced depend on the analysis script and the experiment outputs available when it runs.

## Logging and Reproducibility

The master runner creates a timestamped log under:

`logs/master_run_TIMESTAMP.log`

This log records the master workflow's output, stage execution details, and final summary.

Individual experiment scripts may also create their own logs, CSV files, and plots under `logs/`.

For reproducible comparisons:

- Record the problem sizes, instance count, random seed, and algorithm parameters.
- Preserve the experiment CSV files and logs.
- Compare algorithms using the same test instances wherever possible.
- Distinguish total runtime from average runtime per instance.
- Verify that all items are assigned exactly once and no bin exceeds its capacity.
- Inspect existing experiment outputs before rerunning tests if you need to preserve a particular result set.

## Data and Checkpoint Safety

The master runner expects the checkpoint and dataset at the paths documented above.

Before running experiments, verify that both files exist. The runner's preflight check reports missing files, but individual experiments may still fail if their required assets are unavailable.

**Data generation is opt-in.** The `--generate-data` option executes `utils.py` before the experiments:

```powershell
python .\run_project.py --generate-data
```

Inspect `utils.py` first to understand which files it writes. Do not use this option if you need to preserve existing datasets without confirming that the script will not overwrite them.

## Validation

A valid bin-packing solution must satisfy both conditions:

1. Every input item is assigned to exactly one bin.
2. The total item demand in every bin is no greater than the bin capacity.

Performance comparisons should use validated solutions so that a lower bin count is not accepted at the expense of correctness.

## Troubleshooting

**Missing Python dependencies**

Activate `.venv-deepaco` and install the missing packages in that environment.

**Checkpoint or dataset not found**

Verify the relative locations of `pretrained/bpp/bpp120.pt` and `data/bpp/testDataset-120.pt` relative to the project directory.

**A test stage fails**

Review the error details in the terminal and the timestamped master log under `logs/`. The master runner continues with subsequent stages when an individual experiment fails.

**No new analysis plots appear**

Confirm that the experiment scripts completed successfully and that `analyze_results.py` can find the CSV files it expects.

**Scalability takes too long**

Reduce the number of instances or run fewer problem sizes during debugging. Use the full configuration for the final experiment run.

---

**Project:** DeepACO — Optimized Bin Packing Using Adaptive Item Ordering  
**Language:** Python 3.8  
**Platform:** Windows 10/11, PowerShell  
**Primary objective:** Minimize the number of bins while maintaining valid item assignments and capacity constraints.