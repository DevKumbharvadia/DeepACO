
import sys
import csv
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"
CHECKPOINT_DIR = PROJECT_DIR.parent / "pretrained" / "bpp"

LOG_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Root-level modules in the bpp directory.
from net import Net
from aco import ACO
from utils import gen_pyg_data, load_test_dataset
from local_search import (
    local_search,
    count_bins,
    path_to_bins,
    bin_loads,
)


# ==========================================================
# 2. CONFIGURATION
# ==========================================================

DEVICE = "cpu"
N_NODE = 120
N_ANTS = 20
ACO_ITERATIONS = 20
CAPACITY = 150
EPS = 1e-10


# ==========================================================
# 3. LOGGING
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
# 4. SOLUTION METRICS AND VALIDATION
# ==========================================================

def validate_solution(path, demand, label):
    """
    Validate item assignment and bin capacity.

    Returns:
        bins_used, total_unused_capacity
    """
    if path is None:
        raise RuntimeError(
            f"{label} did not produce a solution."
        )

    bins = path_to_bins(path)
    sizes = demand.detach().cpu().tolist()
    loads = bin_loads(bins, sizes)

    expected_items = list(range(1, len(sizes)))
    actual_items = [
        item
        for bin_items in bins
        for item in bin_items
    ]

    if sorted(actual_items) != sorted(expected_items):
        raise AssertionError(
            f"{label}: item assignment mismatch."
        )

    if any(load > CAPACITY for load in loads):
        raise AssertionError(
            f"{label}: a bin exceeds capacity {CAPACITY}."
        )

    if any(load < 0 for load in loads):
        raise AssertionError(
            f"{label}: invalid negative bin load."
        )

    if len(loads) != count_bins(path):
        raise AssertionError(
            f"{label}: inconsistent bin count."
        )

    unused = sum(CAPACITY - load for load in loads)

    return len(bins), unused


# ==========================================================
# 5. MAIN EXPERIMENT
# ==========================================================

def run_experiment(csv_path):
    checkpoint_path = (
        CHECKPOINT_DIR / f"bpp{N_NODE}.pt"
    ).resolve()

    print("Project directory:", PROJECT_DIR)
    print("Logs directory:", LOG_DIR)
    print("Checkpoint:", checkpoint_path)
    print("Device:", DEVICE)
    print("Number of nodes:", N_NODE)
    print("Number of ants:", N_ANTS)
    print("ACO iterations:", ACO_ITERATIONS)
    print("Bin capacity:", CAPACITY)

    # ------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------

    dataset = load_test_dataset(N_NODE, DEVICE)

    if len(dataset) == 0:
        raise ValueError(
            f"Test dataset for N_NODE={N_NODE} is empty."
        )

    print("Test instances:", len(dataset))

    # ------------------------------------------------------
    # Load pretrained GNN
    # ------------------------------------------------------

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint_path}"
        )

    net = Net().to(DEVICE)

    state = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    net.load_state_dict(state)
    net.eval()

    print("Pretrained checkpoint loaded successfully.")
    print("Starting DeepACO + Local Search evaluation...\n")

    rows = []

    total_deepaco_bins = 0
    total_improved_bins = 0

    total_deepaco_unused = 0
    total_improved_unused = 0

    total_deepaco_time = 0.0
    total_ls_time = 0.0
    improved_instances = 0

    # Aggregate all tracked Local Search statistics.
    aggregate = {
        "move_candidates": 0,
        "accepted_moves": 0,
        "bins_removed_by_moves": 0,
        "swap_candidates": 0,
        "accepted_swaps": 0,
        "bin_elimination_attempts": 0,
        "elimination_placement_candidates": 0,
        "backtracking_nodes": 0,
        "bins_eliminated": 0,
    }

    overall_start = time.perf_counter()

    # ======================================================
    # Evaluate each test instance
    # ======================================================

    for index, demand in enumerate(dataset, start=1):
        instance_start = time.perf_counter()

        # --------------------------------------------------
        # A. DeepACO baseline
        # --------------------------------------------------

        deepaco_start = time.perf_counter()

        with torch.no_grad():
            graph = gen_pyg_data(demand, DEVICE)

            graph.x = graph.x.float()

            heuristic = (
                net(graph).reshape(
                    N_NODE + 1,
                    N_NODE + 1,
                )
                + EPS
            )

            aco = ACO(
                demand=demand,
                n_ants=N_ANTS,
                heuristic=heuristic,
                device=DEVICE,
                capacity=CAPACITY,
            )

            aco.run(ACO_ITERATIONS)

        deepaco_time = (
            time.perf_counter() - deepaco_start
        )

        deepaco_path = aco.shortest_path

        deepaco_bins, deepaco_unused = validate_solution(
            deepaco_path,
            demand,
            "DeepACO",
        )

        # --------------------------------------------------
        # B. Local Search refinement
        # --------------------------------------------------

        stats = {}

        ls_start = time.perf_counter()

        improved_path = local_search(
            deepaco_path,
            demand,
            capacity=CAPACITY,
            stats=stats,
        )

        ls_time = time.perf_counter() - ls_start

        improved_bins, improved_unused = validate_solution(
            improved_path,
            demand,
            "DeepACO+LS",
        )

        # Local Search should not increase the number of bins.
        if improved_bins > deepaco_bins:
            raise AssertionError(
                f"Local Search increased bins for instance {index}: "
                f"{deepaco_bins} -> {improved_bins}"
            )

        # --------------------------------------------------
        # C. Aggregate metrics
        # --------------------------------------------------

        total_deepaco_bins += deepaco_bins
        total_improved_bins += improved_bins

        total_deepaco_unused += deepaco_unused
        total_improved_unused += improved_unused

        total_deepaco_time += deepaco_time
        total_ls_time += ls_time

        bins_saved_this_instance = (
            deepaco_bins - improved_bins
        )

        if bins_saved_this_instance > 0:
            improved_instances += 1

        for key in aggregate:
            aggregate[key] += stats.get(key, 0)

        # --------------------------------------------------
        # D. Record per-instance results
        # --------------------------------------------------

        row = {
            "instance": index,
            "deepaco_bins": deepaco_bins,
            "deepaco_unused_capacity": deepaco_unused,
            "deepaco_runtime_seconds": deepaco_time,
            "local_search_bins": improved_bins,
            "local_search_unused_capacity": improved_unused,
            "local_search_runtime_seconds": ls_time,
            "combined_runtime_seconds": deepaco_time + ls_time,
            "bins_saved": bins_saved_this_instance,
            "valid": True,
        }

        for key in aggregate:
            row[key] = stats.get(key, 0)

        rows.append(row)

        instance_elapsed = time.perf_counter() - instance_start

        print(
            f"Instance {index}/{len(dataset)} | "
            f"DeepACO={deepaco_bins} bins | "
            f"DeepACO+LS={improved_bins} bins | "
            f"Saved={bins_saved_this_instance} | "
            f"Moves={stats.get('move_candidates', 0)} | "
            f"Swaps={stats.get('swap_candidates', 0)} | "
            f"Bins removed by moves="
            f"{stats.get('bins_removed_by_moves', 0)} | "
            f"Bins eliminated="
            f"{stats.get('bins_eliminated', 0)} | "
            f"Elapsed={instance_elapsed:.3f}s"
        )

    total_time = time.perf_counter() - overall_start
    n = len(dataset)

    # ======================================================
    # Save per-instance CSV
    # ======================================================

    if rows:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

    print("\nPer-instance CSV saved:", csv_path)

    # ======================================================
    # Final results
    # ======================================================

    print("\n========== DEEPACO + LOCAL SEARCH ==========")
    print(f"Instances: {n}")

    print(f"DeepACO total bins: {total_deepaco_bins}")
    print(
        f"DeepACO average bins: "
        f"{total_deepaco_bins / n:.3f}"
    )
    print(
        f"DeepACO total unused capacity: "
        f"{total_deepaco_unused}"
    )
    print(
        f"DeepACO average unused capacity: "
        f"{total_deepaco_unused / n:.2f}"
    )
    print(
        f"DeepACO runtime: "
        f"{total_deepaco_time:.3f} seconds"
    )

    print(f"\nDeepACO+LS total bins: {total_improved_bins}")
    print(
        f"DeepACO+LS average bins: "
        f"{total_improved_bins / n:.3f}"
    )
    print(
        f"DeepACO+LS total unused capacity: "
        f"{total_improved_unused}"
    )
    print(
        f"DeepACO+LS average unused capacity: "
        f"{total_improved_unused / n:.2f}"
    )

    print(
        f"Local Search runtime: "
        f"{total_ls_time:.3f} seconds"
    )
    print(
        f"Combined runtime: "
        f"{total_deepaco_time + total_ls_time:.3f} seconds"
    )
    print(
        f"Overall evaluation elapsed time: "
        f"{total_time:.3f} seconds"
    )

    print(f"Improved instances: {improved_instances}")
    print(
        f"Bins saved: "
        f"{total_deepaco_bins - total_improved_bins}"
    )

    # ======================================================
    # Local Search statistics
    # ======================================================

    print("\n========== LOCAL SEARCH CANDIDATE COUNTS ==========")

    for key, value in aggregate.items():
        print(f"{key}: {value}")

    total_candidates = (
        aggregate["move_candidates"]
        + aggregate["swap_candidates"]
        + aggregate["elimination_placement_candidates"]
    )

    print(
        f"Total counted candidate evaluations: "
        f"{total_candidates}"
    )

    # Check whether the counters reconcile with observed savings.
    total_bins_saved = (
        aggregate["bins_removed_by_moves"]
        + aggregate["bins_eliminated"]
    )

    actual_bins_saved = (
        total_deepaco_bins - total_improved_bins
    )

    print(f"Actual bins saved: {actual_bins_saved}")
    print(f"Bins accounted for by counters: {total_bins_saved}")

    if total_bins_saved == actual_bins_saved:
        print("Bin-reduction accounting: PASS")
    else:
        print(
            "WARNING: Bin-reduction counters do not match "
            "the actual reduction. Check statistics tracking."
        )

    print("\nAll evaluated solutions passed validation.")
    print("CSV saved to:", csv_path)


# ==========================================================
# 6. ENTRY POINT AND FILE LOGGING
# ==========================================================

def main():
    # Microseconds reduce the chance of filename collisions.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    log_path = LOG_DIR / (
        f"deepaco_local_search_{stamp}.log"
    )

    csv_path = LOG_DIR / (
        f"deepaco_local_search_{stamp}.csv"
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Save the log beside the CSV, under the project's logs/.
    with log_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as log_file:
        try:
            sys.stdout = Tee(original_stdout, log_file)
            sys.stderr = Tee(original_stderr, log_file)

            print("=" * 60)
            print("DeepACO + Local Search evaluation")
            print(
                "Started:",
                datetime.now().isoformat(timespec="seconds"),
            )
            print("Project directory:", PROJECT_DIR)
            print("Logs directory:", LOG_DIR)
            print("Log file:", log_path)
            print("CSV file:", csv_path)
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

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
