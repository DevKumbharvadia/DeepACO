
"""
DeepACO Master Runner

- Checks Python dependencies before experiments.
- Generates the test dataset automatically only when it is missing.
- Preserves an existing dataset and pretrained checkpoint by default.
- Supports explicit dataset regeneration with --generate-data.
- Continues after individual experiment-stage failures.
- Attempts final analysis after experiment stages.
- Saves master logs under bpp/logs/.
"""

import argparse
import importlib
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"

CHECKPOINT = ROOT.parent / "pretrained" / "bpp" / "bpp120.pt"
DATASET = ROOT.parent / "data" / "bpp" / "testDataset-120.pt"
UTILS_SCRIPT = ROOT / "utils.py"

LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DEPENDENCY CONFIGURATION
# ============================================================

# Python import name -> pip distribution name.
REQUIRED_MODULES = {
    "numpy": "numpy",
    "torch": "torch",
    "scipy": "scipy",
    "networkx": "networkx",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "numba": "numba",
    "torch_geometric": "torch-geometric",
    "pandas": "pandas",
}

# Suggested compatible versions for Python 3.8.
# These are installation recommendations, not a guarantee that
# every project-specific model operation will be compatible.
RECOMMENDED_PACKAGES = [
    "numpy==1.24.4",
    "torch==2.4.1",
    "scipy==1.10.1",
    "networkx==3.1",
    "matplotlib==3.7.5",
    "tqdm==4.66.5",
    "numba==0.58.1",
    "torch-geometric==2.6.1",
    "pandas==2.0.3",
]


# ============================================================
# 3. EXPERIMENT CONFIGURATION
# ============================================================

TEST_STAGES = [
    ("Classical baselines", "test_classical.py"),
    ("Original DeepACO baseline", "test.py"),
    ("DeepACO + Local Search", "test_deepaco_local_search.py"),
    ("Scalability experiments", "test_scalability.py"),
]

ANALYSIS_STAGE = ("Final results analysis", "analyze_results.py")

SKIP_OPTIONS = {
    "test_classical.py": "skip_classical",
    "test.py": "skip_baseline",
    "test_deepaco_local_search.py": "skip_local_search",
    "test_scalability.py": "skip_scalability",
}


# ============================================================
# 4. OUTPUT HELPER
# ============================================================

class Tee:
    """Write output to the terminal and master log simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(
            getattr(stream, "isatty", lambda: False)()
            for stream in self.streams
        )


# ============================================================
# 5. DEPENDENCY CHECK
# ============================================================

def check_dependencies():
    """Check all required imports and report installation instructions."""

    print("\n" + "=" * 76)
    print("DEPENDENCY CHECK")
    print("=" * 76)
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.split()[0])

    if sys.version_info[:2] != (3, 8):
        print(
            "\nWARNING: This project is configured for Python 3.8."
        )

    missing = []

    for module_name, package_name in REQUIRED_MODULES.items():
        try:
            imported = importlib.import_module(module_name)
            version = getattr(imported, "__version__", "version unknown")
            print("[OK]      {} ({})".format(module_name, version))

        except Exception as exc:
            print("[MISSING] {}: {}".format(module_name, exc))
            missing.append(package_name)

    if missing:
        print("\nERROR: Required dependencies are missing or unusable.")
        print("\nInstall the missing packages into this Python environment:")
        print(
            '"{}" -m pip install {}'.format(
                sys.executable,
                " ".join(missing),
            )
        )

        print("\nRecommended pinned environment for Python 3.8:")
        print(
            '"{}" -m pip install {}'.format(
                sys.executable,
                " ".join(RECOMMENDED_PACKAGES),
            )
        )

        print(
            "\nCheck PyTorch/PyTorch Geometric compatibility before "
            "installing or changing versions."
        )
        return False

    print("\nAll required dependencies are importable.")

    try:
        import torch

        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())

        try:
            import torch_geometric

            print(
                "PyTorch Geometric version:",
                torch_geometric.__version__,
            )
        except Exception:
            # Already checked above; retain an explicit diagnostic.
            raise

    except Exception as exc:
        print("ERROR: Framework verification failed:", exc)
        return False

    return True


# ============================================================
# 6. RUN A SCRIPT AND RECORD ITS RESULT
# ============================================================

def run_script(title, filename, results, extra_args=None):
    """
    Execute a script and stream its output to the terminal and master log.
    A failed stage is recorded but does not automatically stop later stages.
    """

    script_path = ROOT / filename
    start_time = time.perf_counter()
    started_at = datetime.now()

    print("\n")
    print("=" * 80)
    print("STAGE:", title)
    print("SCRIPT:", script_path)
    print("STARTED:", started_at.isoformat(timespec="seconds"))
    print("=" * 80)

    command = [sys.executable, str(script_path)]

    if extra_args:
        command.extend(extra_args)

    print("COMMAND:", subprocess.list2cmdline(command))
    print("-" * 80)

    status = "FAILED"
    error_message = ""
    process = None

    if not script_path.is_file():
        error_message = "Script not found: {}".format(script_path)
        print("ERROR:", error_message)

    else:
        try:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end="")

            return_code = process.wait()

            if return_code == 0:
                status = "PASSED"
            else:
                error_message = "Exit code: {}".format(return_code)
                print(
                    "\nERROR: {} exited with code {}.".format(
                        filename,
                        return_code,
                    )
                )

        except KeyboardInterrupt:
            error_message = "Interrupted by user."
            status = "INTERRUPTED"

            if process is not None and process.poll() is None:
                process.terminate()

                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

            print("\nStage interrupted by user.")

        except Exception as exc:
            error_message = "{}: {}".format(type(exc).__name__, exc)
            print("\nERROR:", error_message)
            traceback.print_exc()

        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()

    duration = time.perf_counter() - start_time

    results.append({
        "stage": title,
        "script": filename,
        "status": status,
        "duration_seconds": duration,
        "error": error_message,
    })

    print("-" * 80)
    print("STAGE STATUS:", status)
    print("STAGE RUNTIME: {:.2f} seconds".format(duration))

    if error_message:
        print("DETAIL:", error_message)

    print("=" * 80)

    return status == "PASSED"


# ============================================================
# 7. DATASET GENERATION
# ============================================================

def generate_dataset():
    """
    Run utils.py and verify that the expected dataset is created.

    This function does not run if the existing dataset is being preserved.
    utils.py must save the dataset at DATASET for this check to succeed.
    """

    if not UTILS_SCRIPT.is_file():
        print("ERROR: Cannot generate dataset; utils.py is missing.")
        print("Expected:", UTILS_SCRIPT)
        return False

    print("\nDataset generation started.")
    print("Script:", UTILS_SCRIPT)
    print("Expected output:", DATASET)

    try:
        result = subprocess.run(
            [sys.executable, str(UTILS_SCRIPT)],
            cwd=str(ROOT),
            check=False,
        )
    except Exception as exc:
        print("ERROR: Could not execute utils.py:", exc)
        return False

    if result.returncode != 0:
        print(
            "ERROR: utils.py failed with exit code {}.".format(
                result.returncode
            )
        )
        return False

    if not DATASET.is_file():
        print(
            "ERROR: utils.py finished, but the expected dataset "
            "was not created."
        )
        print("Expected path:", DATASET)
        print("Inspect the output paths in utils.py.")
        return False

    print("Dataset generated successfully.")
    return True


# ============================================================
# 8. PROJECT FILE CHECKS
# ============================================================

def check_project_files(args):
    """
    Validate required scripts and model assets.

    Analysis-only mode does not require the checkpoint or test dataset.
    For experiment modes, a missing dataset is generated automatically.
    """

    print("\n" + "=" * 76)
    print("PROJECT FILE CHECK")
    print("=" * 76)

    required_scripts = [
        ROOT / "utils.py",
        ROOT / "analyze_results.py",
    ]

    if args.analysis_only:
        # Only the analysis script and dependencies are needed here.
        required_scripts = [ROOT / "analyze_results.py"]

    elif args.scalability_only:
        required_scripts.extend([
            ROOT / "test_scalability.py",
        ])

    else:
        required_scripts.extend(
            ROOT / filename for _, filename in TEST_STAGES
        )

    missing_scripts = [
        path for path in required_scripts if not path.is_file()
    ]

    if missing_scripts:
        print("ERROR: Required project scripts are missing:")

        for path in missing_scripts:
            print("  -", path)

        return False

    if args.analysis_only:
        print("Analysis-only mode: experiment assets are not required.")
        print("All required project scripts are available.")
        return True

    # Experiments using the pretrained model require the checkpoint.
    if not CHECKPOINT.is_file():
        print("\nERROR: Pretrained checkpoint not found:")
        print(" ", CHECKPOINT)
        print(
            "Restore the correct checkpoint before running experiments."
        )
        print("The runner will not generate a checkpoint automatically.")
        return False

    if args.generate_data:
        print("\nExplicit --generate-data requested.")
        print(
            "WARNING: utils.py may overwrite files depending on its "
            "implementation."
        )

        if not generate_dataset():
            return False

    elif not DATASET.is_file():
        print("\nTest dataset not found:")
        print(" ", DATASET)
        print("Generating data automatically for the first run...")

        if not generate_dataset():
            print("ERROR: Dataset generation failed; stopping.")
            return False

    else:
        print("\nExisting dataset found; preserving it:")
        print(" ", DATASET)

    if not DATASET.is_file():
        print("\nERROR: Dataset is still missing:")
        print(" ", DATASET)
        return False

    print("\nAll required project files and experiment assets are available.")
    print("Checkpoint:", CHECKPOINT)
    print("Dataset:", DATASET)

    return True


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

def print_summary(results, total_runtime, log_path):
    """Print the final stage statuses and runtimes."""

    passed = sum(r["status"] == "PASSED" for r in results)
    failed = sum(r["status"] == "FAILED" for r in results)
    skipped = sum(r["status"] == "SKIPPED" for r in results)
    interrupted = sum(r["status"] == "INTERRUPTED" for r in results)

    print("\n\n" + "#" * 80)
    print("DEEPACO MASTER RUN - FINAL SUMMARY")
    print("#" * 80)

    print("Project:", ROOT)
    print("Python:", sys.executable)
    print("Finished:", datetime.now().isoformat(timespec="seconds"))

    print("\nSTAGE RESULTS")
    print("-" * 80)
    print("{:<13}{:<34}{:>13}".format(
        "STATUS", "STAGE", "RUNTIME (s)"
    ))
    print("-" * 80)

    for result in results:
        print("{:<13}{:<34}{:>13.2f}".format(
            result["status"],
            result["stage"],
            result["duration_seconds"],
        ))

        if result["error"]:
            print("    Details:", result["error"])

    print("-" * 80)
    print("Passed:      {}".format(passed))
    print("Failed:      {}".format(failed))
    print("Skipped:     {}".format(skipped))
    print("Interrupted: {}".format(interrupted))
    print("Total stages: {}".format(len(results)))
    print("Total workflow runtime: {:.2f} seconds".format(total_runtime))
    print("Master log:", log_path.resolve())

    if failed == 0 and interrupted == 0:
        print("\nRESULT: All attempted stages completed successfully.")
    else:
        print("\nRESULT: The workflow completed with one or more problems.")
        print("Review the failed-stage details and master log.")

    print("#" * 80)


# ============================================================
# 10. WORKFLOW
# ============================================================

def execute_workflow(args, log_path):
    workflow_start = time.perf_counter()
    results = []

    print("=" * 80)
    print("DEEPACO MASTER WORKFLOW")
    print("=" * 80)
    print("Project directory:", ROOT)
    print("Python executable:", sys.executable)
    print("Master log:", log_path.resolve())
    print("Started:", datetime.now().isoformat(timespec="seconds"))

    # Do not run experiments with missing or unusable dependencies.
    if not check_dependencies():
        print("\nPreflight failed: install the required dependencies first.")
        print("No experiment stages were started.")
        return 1

    if not check_project_files(args):
        print("\nPreflight failed: resolve the missing files or assets.")
        print("No experiment stages were started.")
        return 1

    if args.analysis_only:
        run_script(
            ANALYSIS_STAGE[0],
            ANALYSIS_STAGE[1],
            results,
        )

    elif args.scalability_only:
        run_script(
            "Scalability experiments",
            "test_scalability.py",
            results,
            extra_args=[
                "--sizes",
                *[str(size) for size in args.sizes],
                "--instances",
                str(args.instances),
                "--seed",
                str(args.seed),
            ],
        )

        # Attempt analysis even if scalability fails.
        run_script(
            ANALYSIS_STAGE[0],
            ANALYSIS_STAGE[1],
            results,
        )

    else:
        for title, filename in TEST_STAGES:
            skip_option = SKIP_OPTIONS.get(filename)

            if skip_option and getattr(args, skip_option):
                print("\nSKIPPING: {} ({})".format(title, filename))

                results.append({
                    "stage": title,
                    "script": filename,
                    "status": "SKIPPED",
                    "duration_seconds": 0.0,
                    "error": "Skipped by command-line option.",
                })
                continue

            extra_args = None

            if filename == "test_scalability.py":
                extra_args = [
                    "--sizes",
                    *[str(size) for size in args.sizes],
                    "--instances",
                    str(args.instances),
                    "--seed",
                    str(args.seed),
                ]

            # Individual failures are recorded; later stages are attempted.
            run_script(
                title,
                filename,
                results,
                extra_args=extra_args,
            )

        run_script(
            ANALYSIS_STAGE[0],
            ANALYSIS_STAGE[1],
            results,
        )

    total_runtime = time.perf_counter() - workflow_start

    print_summary(results, total_runtime, log_path)

    has_failure = any(
        result["status"] in ("FAILED", "INTERRUPTED")
        for result in results
    )

    return 1 if has_failure else 0


# ============================================================
# 11. ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run DeepACO experiments, analyze results, "
            "and summarize the workflow."
        )
    )

    parser.add_argument(
        "--generate-data",
        action="store_true",
        help=(
            "Explicitly run utils.py, even if the dataset exists. "
            "Inspect utils.py first because it may overwrite data."
        ),
    )

    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Run only analyze_results.py.",
    )

    parser.add_argument(
        "--scalability-only",
        action="store_true",
        help="Run scalability experiments and then final analysis.",
    )

    parser.add_argument(
        "--skip-classical",
        action="store_true",
        help="Skip test_classical.py.",
    )

    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip test.py.",
    )

    parser.add_argument(
        "--skip-local-search",
        action="store_true",
        help="Skip test_deepaco_local_search.py.",
    )

    parser.add_argument(
        "--skip-scalability",
        action="store_true",
        help="Skip test_scalability.py.",
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[60, 90, 120, 150, 180, 210, 240],
        help="Item counts for scalability experiments.",
    )

    parser.add_argument(
        "--instances",
        type=int,
        default=20,
        help="Number of instances per item count.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261017,
        help="Random seed passed to scalability experiments.",
    )

    args = parser.parse_args()

    if args.analysis_only and args.scalability_only:
        parser.error(
            "--analysis-only and --scalability-only cannot be combined."
        )

    if args.generate_data and args.analysis_only:
        parser.error(
            "--generate-data cannot be combined with --analysis-only."
        )

    if any(size <= 0 for size in args.sizes):
        parser.error("All item counts must be positive.")

    if args.instances <= 0:
        parser.error("--instances must be positive.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = LOG_DIR / "master_run_{}.log".format(stamp)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)

        try:
            return execute_workflow(args, log_path)

        except KeyboardInterrupt:
            print("\nMaster workflow interrupted by user.")
            return 130

        except Exception:
            print("\nUNEXPECTED MASTER RUNNER ERROR")
            traceback.print_exc()
            return 1

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            print("\nMaster log saved to:", log_path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
