# DeepACO — Optimized Bin Packing Using Adaptive Item Ordering

## Overview

DeepACO addresses the **Bin Packing Problem (BPP)**: packing items into fixed-capacity bins while minimizing the number of bins used.

The project combines:

- **Graph Neural Network (GNN):** Provides learned guidance for item placement.
- **Ant Colony Optimization (ACO):** Searches for high-quality packing solutions using model-guided decisions.
- **Classical heuristics:** First Fit (FF), First Fit Decreasing (FFD), and Best Fit Decreasing (BFD) provide comparison baselines.
- **Local Search:** Attempts to improve DeepACO solutions through item moves, swaps, and bin elimination.

The project evaluates solution quality, unused capacity, runtime, solution validity, and scalability.

## Requirements

- Windows 10 or Windows 11
- Python 3.8
- PowerShell
- Compatible PyTorch and PyTorch Geometric installations
- Pretrained checkpoint: `../pretrained/bpp/bpp120.pt`
- Test dataset: `../data/bpp/testDataset-120.pt`

Run commands from the `bpp` project directory unless otherwise specified.

The checkpoint and dataset paths are relative to the project directory. The checkpoint must exist before running experiments that require the pretrained model.

## Installation

### 1. Open PowerShell

Navigate to the directory containing the `bpp` project folder.

If you are already in the parent directory, run:

```powershell
cd .\bpp
```

If you are already inside `bpp`, do not run this command again.

### 2. Create and activate a virtual environment

```powershell
py -3.8 -m venv .venv-deepaco
.\.venv-deepaco\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-deepaco\Scripts\Activate.ps1
```

This policy change applies only to the current PowerShell session.

### 3. Install dependencies

Upgrade pip and install the project dependencies:

```powershell
python -m pip install --upgrade "pip<25.1"
python -m pip install numpy==1.24.4 torch==2.4.1 scipy==1.10.1 networkx==3.1 matplotlib==3.7.5 tqdm==4.66.5 numba==0.58.1 torch-geometric==2.6.1 pandas==2.0.3
```

Verify that the selected package versions support Python 3.8 and are compatible with your machine's CPU/GPU configuration.

Official references:

- PyTorch: https://pytorch.org/get-started/previous-versions/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/

### 4. Verify the installation

Check the Python version:

```powershell
python --version
```

Check that the required packages import successfully:

```powershell
python -c "import numpy, torch, scipy, networkx, matplotlib, tqdm, numba, torch_geometric, pandas; print('All dependencies imported successfully')"
```

Check PyTorch and CUDA availability:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Successful imports do not guarantee that every experiment will run successfully.

## How It Works

The project workflow consists of the following stages:

1. Load the pretrained GNN checkpoint and test dataset.
2. Construct bin-packing solutions using GNN-guided ACO.
3. Evaluate classical heuristics as baseline methods.
4. Apply Local Search to attempt to reduce the number of bins.
5. Validate item assignments and bin-capacity constraints.
6. Run scalability experiments across multiple item counts.
7. Save experiment logs and results produced by the individual experiment scripts.

The objective is to minimize the number of bins while maintaining valid item assignments and respecting bin-capacity constraints.

## Run the Project

### Run the complete workflow

From the `bpp` directory, execute:

```powershell
python .\run_project.py
```

The master runner coordinates these experiment stages:

1. Classical baselines — `test_classical.py`
2. Original DeepACO baseline — `test.py`
3. DeepACO with Local Search — `test_deepaco_local_search.py`
4. Scalability experiments — `test_scalability.py`

The default scalability configuration uses seven item counts:

`60, 90, 120, 150, 180, 210, 240`

With 20 instances per size, this corresponds to **140 scalability instances**.

The runner records stage status and runtime, continues to subsequent stages after individual experiment failures, and prints a final execution summary. The summary reports passed, failed, and skipped stages.

A full run can take considerable time. Individual experiment scripts may generate timestamped output files under `logs/`.

### Run scalability experiments only

Run scalability experiments with the default configuration:

```powershell
python .\run_project.py --scalability-only
```

Run all seven sizes with 50 instances per size:

```powershell
python .\run_project.py --scalability-only --sizes 60 90 120 150 180 210 240 --instances 50
```

Run a smaller scalability test:

```powershell
python .\run_project.py --scalability-only --sizes 60 120 240 --instances 10
```

Specify the random seed:

```powershell
python .\run_project.py --scalability-only --seed 20261017
```

These commands assume `run_project.py` accepts and forwards the specified arguments to `test_scalability.py`.

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

Multiple skip options can be combined if supported by the current runner.

### Run individual scripts

You can run each experiment separately:

```powershell
python .\test_classical.py
python .\test.py
python .\test_deepaco_local_search.py
python .\test_scalability.py
```

Use individual scripts when debugging a particular experiment or when you do not need the complete workflow.

## Project Files

| File or directory | Purpose |
|---|---|
| `run_project.py` | Coordinates experiment stages and reports execution status |
| `net.py` | Graph Neural Network model |
| `aco.py` | Ant Colony Optimization implementation |
| `bfd.py` | Best Fit Decreasing heuristic |
| `local_search.py` | Local Search improvement methods |
| `utils.py` | Dataset, graph, and utility functions |
| `test.py` | Original DeepACO baseline evaluation |
| `test_classical.py` | Classical heuristic comparison |
| `test_deepaco_local_search.py` | DeepACO and Local Search comparison |
| `test_scalability.py` | Scalability experiments |
| `train.ipynb` | Notebook for training or experimentation, depending on its implementation |
| `logs/` | Experiment logs, CSV files, plots, and master-run logs |

Keep the existing file structure unchanged unless a deliberate refactor is required.

## Results and Evaluation

The completed experiment run evaluated five methods across seven item counts, using 20 instances per size, for a total of **140 scalability instances**.

### DeepACO vs. DeepACO + Local Search

The recorded paired comparison reported:

| Metric | Result |
|---|---:|
| Improved instances | 33 |
| Unchanged instances | 107 |
| Worsened instances | 0 |
| Total bins saved | 33 |

These values describe the recorded experiment and its tested instances. They do not guarantee identical performance on different datasets, random seeds, problem sizes, or algorithm settings.

Classical heuristics were substantially faster in the completed experiments. Solution quality and runtime should therefore be considered together when comparing methods.

Preserve the experiment CSV files and logs that support these values so the results can be independently verified.

## Logging and Reproducibility

The master runner creates a timestamped log under:

`logs/master_run_TIMESTAMP.log`

The log records workflow output, stage execution details, and the final execution summary. Individual experiment scripts may also create their own logs, CSV files, and plots under `logs/`.

For reproducible comparisons:

- Record problem sizes, instance counts, random seeds, and algorithm parameters.
- Preserve experiment CSV files, logs, and relevant checkpoints.
- Compare algorithms using the same test instances wherever possible.
- Distinguish total runtime from average runtime per instance.
- Validate solutions before comparing their bin counts.
- Inspect existing outputs before rerunning experiments if previous results must be preserved.
- Record dependency versions and the machine configuration used.

A random seed improves reproducibility only when the scripts correctly apply it to all relevant random-number generators and sources of nondeterminism.

## Data and Checkpoint Safety

The master runner expects the following files:

- Pretrained checkpoint: `../pretrained/bpp/bpp120.pt`
- Test dataset: `../data/bpp/testDataset-120.pt`

The pretrained checkpoint must already exist before running experiments that require the trained model. Generating a test dataset does not generate a pretrained checkpoint.

### Dataset generation behavior

The intended behavior is:

- **Dataset missing:** Run `utils.py` to generate the expected dataset.
- **Dataset already exists:** Reuse the existing dataset without regenerating it.
- **Dataset generation fails:** Stop before running experiments that require the missing dataset.
- **Checkpoint missing:** Report the missing checkpoint without attempting to generate one automatically.

This behavior must be implemented in `run_project.py`.

Before relying on automatic generation, inspect `utils.py` to confirm that it writes the dataset to the expected path and determine whether it can overwrite existing files.

If the current runner supports explicit dataset generation, use:

```powershell
python .\run_project.py --generate-data
```

Use this option only when it is implemented by your current runner.

## Validation

A valid bin-packing solution must satisfy both conditions:

1. Every input item is assigned to exactly one bin.
2. The total item demand in every bin does not exceed the bin capacity.

Validate solutions produced by every compared algorithm, not only DeepACO. This ensures that a lower bin count is not accepted at the expense of correctness.

## Troubleshooting

### Missing Python dependencies

Activate the virtual environment:

```powershell
.\.venv-deepaco\Scripts\Activate.ps1
```

Install missing packages into that environment and repeat the import checks.

### PyTorch or PyTorch Geometric installation fails

Confirm that the selected versions support Python 3.8 and are compatible with one another. Follow the official installation instructions rather than repeatedly installing arbitrary versions.

### Checkpoint or dataset not found

Verify the paths relative to the `bpp` directory:

- `..\pretrained\bpp\bpp120.pt`
- `..\data\bpp\testDataset-120.pt`

If the dataset is missing, verify the dataset-generation logic and its output path. If the checkpoint is missing, restore the correct pretrained checkpoint.

### An experiment stage fails

Review the error details in the terminal and the timestamped master log under `logs/`. Check the individual script's inputs, dependencies, and command-line arguments.

### Scalability takes too long

Reduce the number of instances or test fewer problem sizes while debugging. Use the intended full configuration for the final experiment run.

### PowerShell cannot find Python

Check the Python launcher:

```powershell
py -0p
```

Confirm that Python 3.8 is installed and that the virtual environment is active. Then run:

```powershell
python --version
python -m pip --version
```

## Project Summary

- **Project:** DeepACO — Optimized Bin Packing Using Adaptive Item Ordering
- **Language:** Python 3.8
- **Platform:** Windows 10/11, PowerShell
- **Core techniques:** GNN, ACO, classical heuristics, and Local Search
- **Primary objective:** Minimize the number of bins while maintaining valid item assignments and respecting bin-capacity constraints.
