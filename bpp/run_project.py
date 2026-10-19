
"""
DeepACO Master Runner

- Checks dependencies before experiments.
- Generates the dataset only when missing, unless explicitly requested.
- Preserves existing datasets and pretrained checkpoints.
- Continues after individual experiment failures.
- Saves master logs under bpp/logs/.
- Does not run analyze_results.py.
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
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

CHECKPOINT = ROOT.parent / "pretrained" / "bpp" / "bpp120.pt"
DATASET = ROOT.parent / "data" / "bpp" / "testDataset-120.pt"
UTILS = ROOT / "utils.py"

PACKAGES = {
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

STAGES = [
    ("Classical baselines", "test_classical.py", "skip_classical"),
    ("Original DeepACO baseline", "test.py", "skip_baseline"),
    ("DeepACO + Local Search", "test_deepaco_local_search.py", "skip_local_search"),
    ("Scalability experiments", "test_scalability.py", "skip_scalability"),
]


# ============================================================
# LOGGING
# ============================================================

class Tee:
    """Write to the console and log file simultaneously."""

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
# CHECK DEPENDENCIES
# ============================================================

def check_dependencies():
    print("\n" + "=" * 70)
    print("DEPENDENCY CHECK")
    print("=" * 70)
    print("Python:", sys.executable)
    print("Version:", sys.version.split()[0])

    if sys.version_info[:2] != (3, 8):
        print("WARNING: This project is configured for Python 3.8.")

    missing = []

    for module, package in PACKAGES.items():
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "unknown")
            print(f"[OK] {module}: {version}")
        except Exception as exc:
            print(f"[MISSING] {module}: {exc}")
            missing.append(package)

    if missing:
        print("\nInstall missing packages using:")
        print(
            f'"{sys.executable}" -m pip install '
            + " ".join(missing)
        )
        return False

    try:
        import torch
        import torch_geometric

        print("PyTorch:", torch.__version__)
        print("PyTorch Geometric:", torch_geometric.__version__)
        print("CUDA available:", torch.cuda.is_available())
    except Exception as exc:
        print("Framework verification failed:", exc)
        return False

    return True


# ============================================================
# DATASET GENERATION AND FILE CHECKS
# ============================================================

def generate_dataset():
    if not UTILS.is_file():
        print("ERROR: utils.py not found:", UTILS)
        return False

    print("\nGenerating test dataset...")
    result = subprocess.run(
        [sys.executable, str(UTILS)],
        cwd=str(ROOT),
        check=False,
    )

    if result.returncode != 0 or not DATASET.is_file():
        print("ERROR: Dataset generation failed.")
        print("Expected dataset:", DATASET)
        return False

    print("Dataset generated:", DATASET)
    return True


def check_project_files(args):
    required = [UTILS]

    if args.scalability_only:
        required.append(ROOT / "test_scalability.py")
    else:
        required.extend(ROOT / filename for _, filename, _ in STAGES)

    missing = [path for path in required if not path.is_file()]

    if missing:
        print("\nMissing required scripts:")
        for path in missing:
            print(" -", path)
        return False

    if not CHECKPOINT.is_file():
        print("ERROR: Pretrained checkpoint not found:", CHECKPOINT)
        return False

    if args.generate_data:
        print("Explicit dataset generation requested.")
        if not generate_dataset():
            return False
    elif not DATASET.is_file():
        print("Dataset not found; generating it.")
        if not generate_dataset():
            return False
    else:
        print("Using existing dataset:", DATASET)

    return DATASET.is_file()


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_stage(title, filename, results, extra_args=None):
    script = ROOT / filename
    command = [sys.executable, str(script)] + (extra_args or [])
    start = time.perf_counter()

    print("\n" + "=" * 70)
    print("STAGE:", title)
    print("COMMAND:", subprocess.list2cmdline(command))
    print("=" * 70)

    status, error = "FAILED", ""

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

        try:
            if process.stdout:
                for line in process.stdout:
                    print(line, end="")

            code = process.wait()
            if code == 0:
                status = "PASSED"
            else:
                error = f"Exit code: {code}"

        except KeyboardInterrupt:
            status, error = "INTERRUPTED", "Interrupted by user."
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise

        finally:
            if process.stdout:
                process.stdout.close()

    except KeyboardInterrupt:
        print("\nStage interrupted.")
        raise

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    duration = time.perf_counter() - start
    results.append({
        "stage": title,
        "script": filename,
        "status": status,
        "duration_seconds": duration,
        "error": error,
    })

    print(f"\nStatus: {status} | Runtime: {duration:.2f}s")
    if error:
        print("Details:", error)

    return status == "PASSED"


# ============================================================
# WORKFLOW
# ============================================================

def execute(args, log_path):
    start = time.perf_counter()
    results = []

    print("DEEPACO MASTER RUNNER")
    print("Project:", ROOT)
    print("Log:", log_path.resolve())

    if not check_dependencies() or not check_project_files(args):
        print("\nPreflight failed. Experiments were not started.")
        return 1

    if args.scalability_only:
        run_stage(
            "Scalability experiments",
            "test_scalability.py",
            results,
            [
                "--sizes", *map(str, args.sizes),
                "--instances", str(args.instances),
                "--seed", str(args.seed),
            ],
        )
    else:
        for title, filename, skip_flag in STAGES:
            if getattr(args, skip_flag):
                print(f"\nSKIPPED: {title}")
                results.append({
                    "stage": title,
                    "script": filename,
                    "status": "SKIPPED",
                    "duration_seconds": 0.0,
                    "error": "Skipped by command-line option.",
                })
                continue

            extra = None
            if filename == "test_scalability.py":
                extra = [
                    "--sizes", *map(str, args.sizes),
                    "--instances", str(args.instances),
                    "--seed", str(args.seed),
                ]

            run_stage(title, filename, results, extra)

    total = time.perf_counter() - start

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for item in results:
        print(
            f"{item['status']:<12} {item['stage']:<32} "
            f"{item['duration_seconds']:>8.2f}s"
        )
        if item["error"]:
            print("   ", item["error"])

    print("\nPassed:", sum(r["status"] == "PASSED" for r in results))
    print("Failed:", sum(r["status"] == "FAILED" for r in results))
    print("Skipped:", sum(r["status"] == "SKIPPED" for r in results))
    print("Total runtime: {:.2f}s".format(total))
    print("Master log:", log_path.resolve())

    failed = any(
        r["status"] in ("FAILED", "INTERRUPTED")
        for r in results
    )

    print("\nWorkflow finished with errors." if failed
          else "\nAll attempted stages completed successfully.")

    return 1 if failed else 0


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run DeepACO experiments without automatic analysis."
    )

    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Explicitly regenerate the test dataset.",
    )
    parser.add_argument(
        "--scalability-only",
        action="store_true",
        help="Run only scalability experiments.",
    )

    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-local-search", action="store_true")
    parser.add_argument("--skip-scalability", action="store_true")

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[60, 90, 120, 150, 180, 210, 240],
    )
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261017)

    args = parser.parse_args()

    if args.generate_data and args.scalability_only:
        parser.error(
            "--generate-data cannot be combined with --scalability-only."
        )
    if any(size <= 0 for size in args.sizes):
        parser.error("All sizes must be positive.")
    if args.instances <= 0:
        parser.error("--instances must be positive.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = LOGS / f"master_run_{stamp}.log"

    old_stdout, old_stderr = sys.stdout, sys.stderr

    with log_path.open("w", encoding="utf-8", buffering=1) as logfile:
        sys.stdout = Tee(old_stdout, logfile)
        sys.stderr = Tee(old_stderr, logfile)

        try:
            return execute(args, log_path)
        except KeyboardInterrupt:
            print("\nMaster workflow interrupted.")
            return 130
        except Exception:
            traceback.print_exc()
            return 1
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            print("\nMaster log saved to:", log_path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
