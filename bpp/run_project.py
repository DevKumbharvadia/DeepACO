
"""
DeepACO Master Runner

Runs all selected test scripts, continues after individual failures,
analyzes available results, and prints a final execution summary.

All master-run logs are saved under bpp/logs/.
Existing datasets and checkpoints are not generated or overwritten
unless --generate-data is explicitly supplied.
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

LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

REQUIRED_MODULES = [
    "numpy",
    "torch",
    "scipy",
    "networkx",
    "matplotlib",
    "tqdm",
    "numba",
    "torch_geometric",
    "pandas",
]

# All experiment/test scripts, followed by final analysis.
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
# 3. OUTPUT HELPER
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
# 4. PRE-FLIGHT CHECKS
# ============================================================

def check_dependencies():
    """
    Report missing dependencies without stopping the entire workflow.
    Individual scripts will be attempted even if a dependency is missing.
    """

    print("\n" + "=" * 72)
    print("DEPENDENCY CHECK")
    print("=" * 72)

    missing = []

    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
            print(f"[OK]      {module}")
        except Exception as exc:
            print(f"[MISSING] {module}: {exc}")
            missing.append(module)

    if missing:
        print("\nWARNING: Some dependencies are unavailable.")
        print("Install them in the active Python environment:")
        print(
            f'"{sys.executable}" -m pip install '
            + " ".join(missing)
        )
        print("The runner will still attempt the selected test scripts.")
    else:
        print("\nAll required dependencies are available.")

    return not missing



def check_project_files():
    """Check required files and generate data only when needed."""

    print("\n[2/3] Checking project files...")

    utils_script = ROOT / "utils.py"

    # Generate data only if the expected test dataset is missing.
    if not DATASET.is_file():
        if not utils_script.is_file():
            print("ERROR: Dataset is missing and utils.py was not found.")
            return False

        print("\nTest dataset not found:")
        print(" ", DATASET)
        print("Generating data using utils.py...")

        if not run_script("utils.py"):
            print("ERROR: utils.py failed; cannot continue.")
            return False

        if not DATASET.is_file():
            print(
                "ERROR: utils.py completed, but the expected dataset "
                "was not created at:"
            )
            print(" ", DATASET)
            print(
                "Check the output paths and generation logic in utils.py."
            )
            return False

        print("Dataset generated successfully.")

    else:
        print("Existing dataset found; preserving it:")
        print(" ", DATASET)

    # Check the remaining required project files.
    required = [
        ROOT / "aco.py",
        ROOT / "bfd.py",
        ROOT / "local_search.py",
        ROOT / "net.py",
        ROOT / "utils.py",
        ROOT / "test_classical.py",
        ROOT / "test.py",
        ROOT / "test_deepaco_local_search.py",
        ROOT / "test_scalability.py",
        ROOT / "analyze_results.py",
        CHECKPOINT,
        DATASET,
    ]

    missing = [
        path for path in required
        if not path.is_file()
    ]

    if missing:
        print("\nERROR: Required files are missing:")
        for path in missing:
            print("  -", path)

        print(
            "\nRestore missing files or verify the expected paths."
        )
        return False

    print("\nAll required project files are available.")
    print("Dataset:", DATASET)
    print("Checkpoint:", CHECKPOINT)

    return True


# ============================================================
# 5. RUN A SCRIPT AND RECORD ITS RESULT
# ============================================================

def run_script(title, filename, results, extra_args=None):
    """
    Execute one Python script.

    Capture child-process output so it is written to both the terminal
    and the master log. Continue to the next stage after a failure.
    """

    script_path = ROOT / filename
    started_at = datetime.now()
    start = time.perf_counter()

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

    if not script_path.is_file():
        error_message = f"Script not found: {script_path}"
        print("ERROR:", error_message)

    else:
        process = None

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

            # Stream subprocess output to the terminal and master log.
            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end="")

            return_code = process.wait()

            if return_code == 0:
                status = "PASSED"
            else:
                error_message = f"Exit code: {return_code}"
                print(f"\nERROR: {filename} exited with code {return_code}.")

        except KeyboardInterrupt:
            error_message = "Interrupted by user."

            if process is not None and process.poll() is None:
                process.terminate()

                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

            status = "INTERRUPTED"
            print("\nStage interrupted by user.")

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            print("\nERROR:", error_message)
            traceback.print_exc()

        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()

    duration = time.perf_counter() - start

    result = {
        "stage": title,
        "script": filename,
        "status": status,
        "duration_seconds": duration,
        "error": error_message,
    }

    results.append(result)

    print("-" * 80)
    print("STAGE STATUS:", status)
    print(f"STAGE RUNTIME: {duration:.2f} seconds")

    if error_message:
        print("DETAIL:", error_message)

    print("=" * 80)

    return status == "PASSED"


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

def print_summary(results, total_runtime, log_path):
    """Print and save a readable summary of all attempted stages."""

    passed = sum(r["status"] == "PASSED" for r in results)
    failed = sum(r["status"] == "FAILED" for r in results)
    skipped = sum(r["status"] == "SKIPPED" for r in results)
    interrupted = sum(r["status"] == "INTERRUPTED" for r in results)

    print("\n\n")
    print("#" * 80)
    print("DEEPACO MASTER RUN — FINAL SUMMARY")
    print("#" * 80)

    print("Project:", ROOT)
    print("Python:", sys.executable)
    print("Finished:", datetime.now().isoformat(timespec="seconds"))

    print("\nSTAGE RESULTS")
    print("-" * 80)

    print(
        f"{'STATUS':<13}"
        f"{'STAGE':<34}"
        f"{'RUNTIME (s)':>13}"
    )
    print("-" * 80)

    for result in results:
        print(
            f"{result['status']:<13}"
            f"{result['stage']:<34}"
            f"{result['duration_seconds']:>13.2f}"
        )

        if result["error"]:
            print(f"    Details: {result['error']}")

    print("-" * 80)
    print(f"Passed:     {passed}")
    print(f"Failed:     {failed}")
    print(f"Skipped:    {skipped}")
    print(f"Interrupted:{interrupted:>4}")
    print(f"Total stages: {len(results)}")
    print(f"Total workflow runtime: {total_runtime:.2f} seconds")
    print(f"Master log: {log_path.resolve()}")

    if failed == 0 and interrupted == 0:
        print("\nRESULT: All attempted stages completed successfully.")
    else:
        print("\nRESULT: The workflow completed with one or more problems.")
        print("Review the failed-stage details and master log.")

    print("#" * 80)


# ============================================================
# 7. WORKFLOW
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

    check_dependencies()
    check_project_files()

    # Optional data-generation stage.
    if args.generate_data:
        run_script(
            "Generate test data",
            "utils.py",
            results,
        )

    # Analysis-only mode.
    if args.analysis_only:
        run_script(
            ANALYSIS_STAGE[0],
            ANALYSIS_STAGE[1],
            results,
        )

    # Scalability-only mode, followed by analysis.
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

        # Always attempt analysis, even if scalability failed.
        run_script(
            ANALYSIS_STAGE[0],
            ANALYSIS_STAGE[1],
            results,
        )

    # Default: run all selected test scripts, then analyze.
    else:
        for title, filename in TEST_STAGES:
            skip_option = SKIP_OPTIONS.get(filename)

            if skip_option and getattr(args, skip_option):
                print(f"\nSKIPPING: {title} ({filename})")
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

            # Failure in one stage does not stop later stages.
            run_script(
                title,
                filename,
                results,
                extra_args=extra_args,
            )

        # Always attempt final analysis after the test stages.
        run_script(
            ANALYSIS_STAGE[0],
            ANALYSIS_STAGE[1],
            results,
        )

    total_runtime = time.perf_counter() - workflow_start

    print_summary(
        results,
        total_runtime,
        log_path,
    )

    has_failure = any(
        r["status"] in ("FAILED", "INTERRUPTED")
        for r in results
    )

    return 1 if has_failure else 0


# ============================================================
# 8. ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run DeepACO tests, collect their results, "
            "and summarize the workflow."
        )
    )

    parser.add_argument(
        "--generate-data",
        action="store_true",
        help=(
            "Run utils.py before the tests. Inspect its write behavior "
            "before using this option."
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
        help="Random seed for scalability experiments.",
    )

    args = parser.parse_args()

    if args.analysis_only and args.scalability_only:
        parser.error(
            "--analysis-only and --scalability-only cannot be combined."
        )

    if any(size <= 0 for size in args.sizes):
        parser.error("All item counts must be positive.")

    if args.instances <= 0:
        parser.error("--instances must be positive.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = LOG_DIR / f"master_run_{stamp}.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with log_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as log_file:
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
