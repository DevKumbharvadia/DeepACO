
import sys
import csv
import time
import traceback
import torch
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

# ==========================================================
# 1. PROJECT PATHS AND IMPORTS
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import load_test_dataset
from bfd import best_fit_decreasing


# ==========================================================
# 2. CONFIGURATION
# ==========================================================

CAPACITY = 150
N_INSTANCES = 100
PROBLEM_SIZE = 120


# ==========================================================
# 3. LOGGING HELPER
# ==========================================================

class Tee:
    """Write output to both the terminal and a log file."""

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


# ==========================================================
# 4. DATA AND CLASSICAL ALGORITHMS
# ==========================================================

def get_items(instance):
    """Remove the dummy depot at index zero."""

    if hasattr(instance, "detach"):
        instance = instance.detach().cpu().tolist()

    return [int(x) for x in instance[1:]]


def first_fit(items, capacity=CAPACITY):
    """First Fit: place each item in the first feasible bin."""

    bins = []
    loads = []

    for item in items:
        if item < 0 or item > capacity:
            raise ValueError(
                f"Invalid item demand {item} for capacity {capacity}"
            )

        placed = False

        for i, load in enumerate(loads):
            if load + item <= capacity:
                bins[i].append(item)
                loads[i] += item
                placed = True
                break

        if not placed:
            bins.append([item])
            loads.append(item)

    return bins


def first_fit_decreasing(items, capacity=CAPACITY):
    """First Fit Decreasing: sort descending, then use First Fit."""

    return first_fit(
        sorted(items, reverse=True),
        capacity,
    )


def bin_stats(bins, capacity=CAPACITY):
    """Return bin count and total unused capacity."""

    loads = [sum(bin_items) for bin_items in bins]

    if any(load > capacity for load in loads):
        raise AssertionError("A bin exceeds the capacity.")

    unused = sum(capacity - load for load in loads)

    return len(bins), unused


def validate_solution(bins, original_items, method):
    """Ensure the algorithm preserves all items and respects capacity."""

    flattened = [
        item
        for bin_items in bins
        for item in bin_items
    ]

    if sorted(flattened) != sorted(original_items):
        raise AssertionError(
            f"{method}: item assignment mismatch."
        )

    if any(
        sum(bin_items) > CAPACITY
        for bin_items in bins
    ):
        raise AssertionError(
            f"{method}: bin capacity exceeded."
        )


# ==========================================================
# 5. EXPERIMENT
# ==========================================================

def run_experiment(csv_path):
    print("Project directory:", PROJECT_DIR)
    print("Logs directory:", LOG_DIR)
    print("Problem size:", PROBLEM_SIZE)
    print("Requested instances:", N_INSTANCES)
    print("Bin capacity:", CAPACITY)

    dataset = load_test_dataset(PROBLEM_SIZE, "cpu")

    if dataset is None:
        raise ValueError(
            f"The test dataset for size {PROBLEM_SIZE} is missing."
        )

    if isinstance(dataset, torch.Tensor) and dataset.numel() == 0:
        raise ValueError(
            f"The test dataset for size {PROBLEM_SIZE} is empty."
        )

    if isinstance(dataset, (list, tuple, dict, str)) and len(dataset) == 0:
        raise ValueError(
            f"The test dataset for size {PROBLEM_SIZE} is empty."
        )

    if len(dataset) < N_INSTANCES:
        print(
            f"WARNING: Requested {N_INSTANCES} instances, "
            f"but the dataset contains only {len(dataset)}. "
            "Evaluating all available instances."
        )

    dataset = dataset[:N_INSTANCES]

    methods = {
        "FF": lambda items: first_fit(items, CAPACITY),
        "FFD": lambda items: first_fit_decreasing(items, CAPACITY),
        "BFD": lambda items: best_fit_decreasing(
            [0] + items,
            CAPACITY,
        ),
    }

    results = {
        name: []
        for name in methods
    }

    elapsed = {
        name: 0.0
        for name in methods
    }

    print(f"\nEvaluating {len(dataset)} instances...\n")

    for instance_id, instance in enumerate(dataset, start=1):
        items = get_items(instance)

        for name, method in methods.items():
            start = time.perf_counter()

            bins = method(items)

            duration = time.perf_counter() - start
            elapsed[name] += duration

            validate_solution(bins, items, name)

            count, unused = bin_stats(bins, CAPACITY)

            results[name].append({
                "instance": instance_id,
                "bins": count,
                "unused_capacity": unused,
                "runtime_seconds": duration,
                "valid": True,
            })

        print(
            f"Instance {instance_id}/{len(dataset)} | "
            f"FF={results['FF'][-1]['bins']} bins | "
            f"FFD={results['FFD'][-1]['bins']} bins | "
            f"BFD={results['BFD'][-1]['bins']} bins"
        )

    # ======================================================
    # 6. SUMMARY
    # ======================================================

    print("\n========== CLASSICAL BASELINE RESULTS ==========")
    print(f"Instances: {len(dataset)}")
    print(f"Bin capacity: {CAPACITY}")

    for name in methods:
        rows = results[name]
        n = len(rows)

        total_bins = sum(
            row["bins"] for row in rows
        )

        total_unused = sum(
            row["unused_capacity"] for row in rows
        )

        print(f"\n--- {name} ---")
        print(f"Total bins: {total_bins}")
        print(f"Average bins: {total_bins / n:.3f}")
        print(f"Total unused capacity: {total_unused}")
        print(
            f"Average unused capacity per instance: "
            f"{total_unused / n:.2f}"
        )
        print(f"Total runtime: {elapsed[name]:.6f} seconds")
        print(
            f"Average runtime per instance: "
            f"{elapsed[name] / n:.6f} seconds"
        )

    # ======================================================
    # 7. SAVE PER-INSTANCE CSV
    # ======================================================

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "instance",
            "FF_bins",
            "FF_unused",
            "FF_runtime_seconds",
            "FFD_bins",
            "FFD_unused",
            "FFD_runtime_seconds",
            "BFD_bins",
            "BFD_unused",
            "BFD_runtime_seconds",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for i in range(len(dataset)):
            row = {
                "instance": i + 1,
            }

            for name in ("FF", "FFD", "BFD"):
                result = results[name][i]

                row[f"{name}_bins"] = result["bins"]
                row[f"{name}_unused"] = result["unused_capacity"]
                row[f"{name}_runtime_seconds"] = (
                    result["runtime_seconds"]
                )

            writer.writerow(row)

    print("\nPer-instance CSV saved to:", csv_path.resolve())
    print("All classical solutions passed validation.")


# ==========================================================
# 8. ENTRY POINT AND LOGGING
# ==========================================================

def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    log_path = LOG_DIR / (
        f"classical_comparison_{stamp}.log"
    )

    csv_path = LOG_DIR / (
        f"classical_comparison_{stamp}.csv"
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with log_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as log_file:
        try:
            sys.stdout = Tee(original_stdout, log_file)
            sys.stderr = Tee(original_stderr, log_file)

            print("=" * 60)
            print("Classical bin-packing baseline evaluation")
            print(
                "Started:",
                datetime.now().isoformat(timespec="seconds"),
            )
            print("Log file:", log_path.resolve())
            print("CSV file:", csv_path.resolve())
            print("=" * 60)

            try:
                run_experiment(csv_path)
            except Exception:
                print("\nEXPERIMENT FAILED")
                traceback.print_exc()
                raise

            print(
                "\nFinished:",
                datetime.now().isoformat(timespec="seconds"),
            )
            print("Experiment completed successfully.")

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print("Log saved to:", log_path.resolve())


if __name__ == "__main__":
    main()
