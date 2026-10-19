
import os
import sys
import csv
import time
import random
import argparse
import statistics
import traceback
from datetime import datetime
from collections import Counter
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================
# 1. PROJECT PATHS AND IMPORTS
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent
LOG_ROOT = PROJECT_DIR / "logs"
CHECKPOINT_DIR = PROJECT_DIR.parent / "pretrained" / "bpp"

LOG_ROOT.mkdir(parents=True, exist_ok=True)

# Make root-level project modules importable when this script
# is launched directly from PowerShell or another directory.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from net import Net
from aco import ACO
from utils import gen_pyg_data
from local_search import local_search, path_to_bins, bin_loads

try:
    import psutil
except ImportError:
    psutil = None


# ==========================================================
# 2. EXPERIMENT CONFIGURATION
# ==========================================================

CAPACITY = 150
N_ANTS = 20
ACO_ITERATIONS = 20
EPS = 1e-10
DEVICE = "cpu"

METHODS = ["FF", "FFD", "BFD", "DeepACO", "DeepACO+LS"]


# ==========================================================
# 3. LOGGING HELPER
# ==========================================================

class Tee:
    """Write output to both the terminal and a log file."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()


# ==========================================================
# 4. RESOURCE MEASUREMENT
# ==========================================================

def memory_mb():
    """Return current process RSS in MB, if psutil is available."""
    if psutil is None:
        return None

    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


# ==========================================================
# 5. CLASSICAL BIN-PACKING ALGORITHMS
# ==========================================================

def classical_pack(items, method):
    """
    Pack items and count bin-fit comparisons in the same execution.

    FF:
        First Fit, original item order.

    FFD:
        First Fit Decreasing.

    BFD:
        Best Fit Decreasing.

    Candidate counts are comparisons against existing bins.
    """

    if method not in ("FF", "FFD", "BFD"):
        raise ValueError("Unknown classical method: " + method)

    if method in ("FFD", "BFD"):
        ordered = sorted(items, reverse=True)
    else:
        ordered = list(items)

    bins = []
    loads = []
    candidates = 0

    for item in ordered:
        chosen = -1

        for i, load in enumerate(loads):
            candidates += 1

            if load + item <= CAPACITY:
                if method in ("FF", "FFD"):
                    chosen = i
                    break

                # BFD chooses the feasible bin with the least
                # remaining capacity after placement.
                if chosen == -1 or loads[i] > loads[chosen]:
                    chosen = i

        if chosen == -1:
            bins.append([item])
            loads.append(item)
        else:
            bins[chosen].append(item)
            loads[chosen] += item

    return bins, candidates


# ==========================================================
# 6. SOLUTION VALIDATION
# ==========================================================

def validate_classical(bins, items):
    """Check item preservation and bin capacity."""

    flattened = [
        item
        for bin_items in bins
        for item in bin_items
    ]

    assert Counter(flattened) == Counter(items), (
        "Classical algorithm changed or lost items"
    )

    assert all(
        sum(bin_items) <= CAPACITY
        for bin_items in bins
    ), "Classical algorithm exceeded bin capacity"

    unused = sum(
        CAPACITY - sum(bin_items)
        for bin_items in bins
    )

    return len(bins), unused


def validate_path(path, demand):
    """Validate a DeepACO path and calculate bin count and unused capacity."""

    if path is None:
        raise RuntimeError("DeepACO did not produce a solution.")

    bins = path_to_bins(path)
    sizes = demand.detach().cpu().tolist()

    loads = bin_loads(bins, sizes)

    expected_items = list(range(1, len(sizes)))
    actual_items = [
        item
        for bin_items in bins
        for item in bin_items
    ]

    assert Counter(actual_items) == Counter(expected_items), (
        "DeepACO item assignment mismatch"
    )

    assert all(
        load <= CAPACITY for load in loads
    ), "DeepACO exceeded bin capacity"

    unused = sum(
        CAPACITY - load for load in loads
    )

    return len(bins), unused


# ==========================================================
# 7. DETERMINISTIC INSTANCE GENERATION
# ==========================================================

def generate_instances(size, count, seed):
    """
    Generate deterministic instances in memory.

    Element 0 is the dummy depot.
    Item demands are integers from 1 to 100.

    These are generated scalability instances, not the fixed test dataset.
    """

    rng = random.Random(seed + size * 100003)

    return [
        [0] + [
            rng.randint(1, 100)
            for _ in range(size)
        ]
        for _ in range(count)
    ]


# ==========================================================
# 8. MODEL AND ACO WARM-UP
# ==========================================================

def warmup_model_and_aco(net):
    """Warm up model inference and ACO before measured runs."""

    print("Warming up GNN inference and ACO...")

    warmup_demand = torch.tensor(
        [0] + [20] * 10,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        graph = gen_pyg_data(warmup_demand, DEVICE)

        heuristic = (
            net(graph).reshape(
                len(warmup_demand),
                len(warmup_demand),
            )
            + EPS
        )

    warmup_aco = ACO(
        demand=warmup_demand,
        n_ants=N_ANTS,
        heuristic=heuristic,
        device=DEVICE,
        capacity=CAPACITY,
    )

    warmup_aco.run(1)

    print("Warm-up complete; measured runs start now.")


# ==========================================================
# 9. DEEPACO EXECUTION
# ==========================================================

def run_deepaco(demand, net, seed):
    """Run DeepACO and return timing, quality, and resource measurements."""

    torch.manual_seed(seed)

    rss_before = memory_mb()

    # Measure GNN inference separately.
    start = time.perf_counter()

    with torch.no_grad():
        graph = gen_pyg_data(demand, DEVICE)

        heuristic = (
            net(graph).reshape(
                len(demand),
                len(demand),
            )
            + EPS
        )

    inference_seconds = time.perf_counter() - start

    # Measure ACO separately.
    start = time.perf_counter()

    aco = ACO(
        demand=demand,
        n_ants=N_ANTS,
        heuristic=heuristic,
        device=DEVICE,
        capacity=CAPACITY,
    )

    aco.run(ACO_ITERATIONS)

    aco_seconds = time.perf_counter() - start

    path = aco.shortest_path

    bins, unused = validate_path(path, demand)

    rss_after = memory_mb()

    return {
        "bins": bins,
        "unused_capacity": unused,
        "inference_seconds": inference_seconds,
        "aco_seconds": aco_seconds,
        "runtime_seconds": inference_seconds + aco_seconds,

        # This is an estimate of generated ant paths, not individual
        # item-to-bin scores or fitness-function evaluations.
        "candidate_evaluations": N_ANTS * ACO_ITERATIONS,

        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "path": path,
    }


# ==========================================================
# 10. RUN SCALABILITY EXPERIMENT
# ==========================================================

def run_experiment(sizes, instances, seed, output_dir):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        CHECKPOINT_DIR / "bpp120.pt"
    ).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing pretrained checkpoint: {checkpoint_path}"
        )

    print("Project directory:", PROJECT_DIR)
    print("Output directory:", output_dir)
    print("Loading unchanged pretrained checkpoint...")
    print("Checkpoint:", checkpoint_path)

    net = Net().to(DEVICE)

    state = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    net.load_state_dict(state)
    net.eval()

    warmup_model_and_aco(net)

    rows = []

    for size in sizes:
        print(f"\n===== SIZE: {size} ITEMS =====")

        generated = generate_instances(
            size,
            instances,
            seed,
        )

        for index, values in enumerate(generated, start=1):
            demand = torch.tensor(
                values,
                dtype=torch.float32,
                device=DEVICE,
            )

            items = values[1:]

            instance_seed = (
                seed + size * 100003 + index
            )

            print(f"\nInstance {index}/{instances}")

            # --------------------------------------------------
            # A. CLASSICAL METHODS
            # --------------------------------------------------

            for method in ("FF", "FFD", "BFD"):
                rss_before = memory_mb()

                start = time.perf_counter()

                bins, candidates = classical_pack(
                    items,
                    method,
                )

                elapsed = time.perf_counter() - start

                rss_after = memory_mb()

                bin_count, unused = validate_classical(
                    bins,
                    items,
                )

                rows.append({
                    "size": size,
                    "instance": index,
                    "seed": seed,
                    "instance_seed": instance_seed,
                    "method": method,
                    "bins": bin_count,
                    "unused_capacity": unused,
                    "runtime_seconds": elapsed,
                    "inference_seconds": 0.0,
                    "aco_seconds": 0.0,
                    "local_search_seconds": 0.0,
                    "candidate_evaluations": candidates,
                    "rss_before_mb": rss_before,
                    "rss_after_mb": rss_after,
                    "valid": True,
                })

                print(
                    f"  {method}: {bin_count} bins, "
                    f"{elapsed:.4f}s, "
                    f"{candidates} bin-fit comparisons"
                )

            # --------------------------------------------------
            # B. DEEPACO BASELINE
            # --------------------------------------------------

            deep = run_deepaco(
                demand,
                net,
                instance_seed,
            )

            # Preserve the original path for the LS experiment.
            path = deep.pop("path")

            rows.append({
                "size": size,
                "instance": index,
                "seed": seed,
                "instance_seed": instance_seed,
                "method": "DeepACO",
                "bins": deep["bins"],
                "unused_capacity": deep["unused_capacity"],
                "runtime_seconds": deep["runtime_seconds"],
                "inference_seconds": deep["inference_seconds"],
                "aco_seconds": deep["aco_seconds"],
                "local_search_seconds": 0.0,
                "candidate_evaluations": deep["candidate_evaluations"],
                "rss_before_mb": deep["rss_before_mb"],
                "rss_after_mb": deep["rss_after_mb"],
                "valid": True,
            })

            # --------------------------------------------------
            # C. DEEPACO + LOCAL SEARCH
            # --------------------------------------------------

            stats = {}

            rss_before = memory_mb()

            start = time.perf_counter()

            improved_path = local_search(
                path,
                demand,
                capacity=CAPACITY,
                stats=stats,
            )

            ls_seconds = time.perf_counter() - start

            rss_after = memory_mb()

            improved_bins, improved_unused = validate_path(
                improved_path,
                demand,
            )

            assert improved_bins <= deep["bins"], (
                "Local Search increased the bin count"
            )

            ls_candidates = (
                stats.get("move_candidates", 0)
                + stats.get("swap_candidates", 0)
                + stats.get(
                    "elimination_placement_candidates",
                    0,
                )
            )

            rows.append({
                "size": size,
                "instance": index,
                "seed": seed,
                "instance_seed": instance_seed,
                "method": "DeepACO+LS",
                "bins": improved_bins,
                "unused_capacity": improved_unused,

                # End-to-end runtime includes the same DeepACO run
                # plus the additional local-search refinement.
                "runtime_seconds": (
                    deep["runtime_seconds"] + ls_seconds
                ),

                "inference_seconds": deep["inference_seconds"],
                "aco_seconds": deep["aco_seconds"],
                "local_search_seconds": ls_seconds,

                "candidate_evaluations": (
                    deep["candidate_evaluations"] + ls_candidates
                ),

                "rss_before_mb": rss_before,
                "rss_after_mb": rss_after,
                "valid": True,

                "move_candidates": stats.get(
                    "move_candidates", 0
                ),
                "swap_candidates": stats.get(
                    "swap_candidates", 0
                ),
                "elimination_placement_candidates": stats.get(
                    "elimination_placement_candidates", 0
                ),
            })

            print(
                f"  DeepACO: {deep['bins']} bins, "
                f"{deep['runtime_seconds']:.3f}s | "
                f"DeepACO+LS: {improved_bins} bins, "
                f"{ls_seconds:.3f}s refinement"
            )

    # ======================================================
    # 11. SAVE PER-INSTANCE RESULTS
    # ======================================================

    raw_path = output_dir / "scalability_per_instance.csv"

    fields = sorted({
        key
        for row in rows
        for key in row
    })

    with raw_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nPer-instance results saved:", raw_path)

    # ======================================================
    # 12. AGGREGATE SUMMARY
    # ======================================================

    summary = []

    for size in sizes:
        for method in METHODS:
            group = [
                row
                for row in rows
                if row["size"] == size
                and row["method"] == method
            ]

            if not group:
                continue

            rss_values = [
                row["rss_after_mb"]
                for row in group
                if row["rss_after_mb"] is not None
            ]

            summary.append({
                "size": size,
                "method": method,
                "instances": len(group),

                "average_bins": statistics.mean(
                    row["bins"] for row in group
                ),

                "total_bins": sum(
                    row["bins"] for row in group
                ),

                "average_unused_capacity": statistics.mean(
                    row["unused_capacity"] for row in group
                ),

                "total_unused_capacity": sum(
                    row["unused_capacity"] for row in group
                ),

                "total_runtime_seconds": sum(
                    row["runtime_seconds"] for row in group
                ),

                "average_runtime_seconds": statistics.mean(
                    row["runtime_seconds"] for row in group
                ),

                "total_candidate_evaluations": sum(
                    row["candidate_evaluations"]
                    for row in group
                ),

                "average_rss_after_mb": (
                    statistics.mean(rss_values)
                    if rss_values
                    else None
                ),
            })

    summary_path = output_dir / "scalability_summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary)

    print("Summary results saved:", summary_path)

    # ======================================================
    # 13. GENERATE PLOTS
    # ======================================================

    make_plots(summary, output_dir)

    print("\n===== EXPERIMENT COMPLETE =====")
    print("Per-instance CSV:", raw_path)
    print("Summary CSV:", summary_path)
    print("Graphs directory:", output_dir.resolve())

    if psutil is None:
        print(
            "Memory measurements unavailable: "
            "optional psutil package is not installed."
        )

    print(
        "All recorded solutions passed assignment "
        "and capacity validation."
    )


# ==========================================================
# 14. PLOT GENERATION
# ==========================================================

def make_plots(summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        (
            "average_bins",
            "Average bins per instance",
            "average_bins.png",
        ),
        (
            "average_unused_capacity",
            "Average unused capacity",
            "average_unused_capacity.png",
        ),
        (
            "average_runtime_seconds",
            "Average runtime (seconds)",
            "average_runtime.png",
        ),
        (
            "total_candidate_evaluations",
            "Total candidate evaluations",
            "candidate_evaluations.png",
        ),
    ]

    for field, ylabel, filename in metrics:
        plt.figure(figsize=(8, 5))

        for method in METHODS:
            group = sorted(
                (
                    row
                    for row in summary
                    if row["method"] == method
                ),
                key=lambda row: row["size"],
            )

            if group:
                plt.plot(
                    [row["size"] for row in group],
                    [row[field] for row in group],
                    marker="o",
                    label=method,
                )

        plt.xlabel("Number of items")
        plt.ylabel(ylabel)
        plt.title(ylabel + " vs. problem size")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_path = output_dir / filename

        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print("Plot saved:", plot_path)


# ==========================================================
# 15. COMMAND-LINE ENTRY POINT AND LOGGING
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description="DeepACO bin-packing scalability experiment"
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[60, 90, 120, 150],
        help="Problem sizes, measured as number of items.",
    )

    parser.add_argument(
        "--instances",
        type=int,
        default=10,
        help="Number of generated instances per problem size.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261017,
        help="Random seed for reproducible instance generation.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output directory; relative paths resolve from bpp/.",
    )

    args = parser.parse_args()

    if args.instances < 1:
        parser.error("--instances must be at least 1")

    if any(size < 1 for size in args.sizes):
        parser.error("All problem sizes must be positive")

    if len(set(args.sizes)) != len(args.sizes):
        parser.error("Problem sizes must not contain duplicates")

    # Microseconds reduce the chance of output collisions when
    # multiple experiments start within the same second.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if args.output:
        output_dir = Path(args.output)

        if not output_dir.is_absolute():
            output_dir = PROJECT_DIR / output_dir
    else:
        output_dir = LOG_ROOT / f"scalability_{stamp}"

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"scalability_{stamp}.log"

    # Keep stdout/stderr visible while duplicating output to disk.
    # Line buffering plus explicit flushes helps preserve progress
    # if the experiment is interrupted.
    with log_path.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as log_file:

        stdout_tee = Tee(sys.stdout, log_file)
        stderr_tee = Tee(sys.stderr, log_file)

        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            print("=" * 65)
            print("DeepACO scalability experiment")
            print(
                "Started:",
                datetime.now().isoformat(timespec="seconds"),
            )
            print("Project directory:", PROJECT_DIR)
            print("Output directory:", output_dir)
            print("Log file:", log_path)
            print("Sizes:", args.sizes)
            print("Instances per size:", args.instances)
            print("Seed:", args.seed)
            print("Capacity:", CAPACITY)
            print("Device:", DEVICE)
            print("Ants:", N_ANTS)
            print("ACO iterations:", ACO_ITERATIONS)
            print("=" * 65)

            try:
                run_experiment(
                    sizes=args.sizes,
                    instances=args.instances,
                    seed=args.seed,
                    output_dir=output_dir,
                )

                print(
                    "Finished:",
                    datetime.now().isoformat(timespec="seconds"),
                )
                print("Experiment completed successfully.")

            except Exception:
                print("\nEXPERIMENT FAILED")
                traceback.print_exc()
                raise SystemExit(1)


if __name__ == "__main__":
    main()
