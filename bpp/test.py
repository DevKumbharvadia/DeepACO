
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# --------------------------------------------------
# 1. Resolve paths from this script's location
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"

# Based on the expected layout:
# DeepACO/
#   pretrained/bpp/bpp120.pt
#   bpp/
#     test.py
#     aco.py
#     net.py
#     utils.py
CHECKPOINT_DIR = PROJECT_DIR.parent / "pretrained" / "bpp"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# Make imports work when this script is run directly
# from the bpp directory.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from net import Net
from aco import ACO
from utils import gen_pyg_data, load_test_dataset


# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------
DEVICE = "cpu"
N_NODE = 120
N_ANTS = 20
ACO_ITERATIONS = 20
EPS = 1e-10


def main():
    # Unique filenames prevent overwriting earlier results.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = LOG_DIR / f"deepaco_evaluation_{timestamp}.log"
    csv_path = LOG_DIR / f"deepaco_evaluation_{timestamp}.csv"

    # Log to both the terminal and a file in logs/.
    logger = logging.getLogger("deepaco_evaluation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_path, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    try:
        logger.info("Starting DeepACO evaluation")
        logger.info("Project directory: %s", PROJECT_DIR)
        logger.info("Logs directory: %s", LOG_DIR)
        logger.info("Device: %s", DEVICE)
        logger.info("Number of nodes: %s", N_NODE)
        logger.info("Number of ants: %s", N_ANTS)
        logger.info("ACO iterations: %s", ACO_ITERATIONS)

        # --------------------------------------------------
        # 3. Load test dataset
        # --------------------------------------------------
        test_list = load_test_dataset(N_NODE, DEVICE)

        if len(test_list) == 0:
            raise ValueError("The test dataset is empty.")

        logger.info("Test instances: %s", len(test_list))

        # --------------------------------------------------
        # 4. Load pretrained GNN
        # --------------------------------------------------
        net = Net().to(DEVICE)

        checkpoint_path = (
            CHECKPOINT_DIR / f"bpp{N_NODE}.pt"
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )

        net.load_state_dict(checkpoint)
        net.eval()

        logger.info("Checkpoint: %s", checkpoint_path)
        logger.info("Starting instance evaluation")

        # --------------------------------------------------
        # 5. Evaluate instances and save each result to CSV
        # --------------------------------------------------
        total_bins = 0
        instance_results = []
        overall_start = time.perf_counter()

        with torch.no_grad():
            for index, demand in enumerate(test_list, start=1):
                instance_start = time.perf_counter()

                graph = gen_pyg_data(demand, DEVICE)

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
                )

                aco.run(ACO_ITERATIONS)

                if aco.shortest_path is None:
                    raise RuntimeError(
                        f"ACO did not produce a solution for "
                        f"instance {index}."
                    )

                bins_used = int(
                    aco.shortest_path.eq(0).sum().item()
                )

                instance_elapsed = (
                    time.perf_counter() - instance_start
                )

                total_bins += bins_used

                instance_results.append({
                    "instance": index,
                    "bins_used": bins_used,
                    "runtime_seconds": round(
                        instance_elapsed, 6
                    ),
                })

                logger.info(
                    "Instance %s/%s | bins=%s | runtime=%.3f sec",
                    index,
                    len(test_list),
                    bins_used,
                    instance_elapsed,
                )

        elapsed = time.perf_counter() - overall_start
        average_bins = total_bins / len(test_list)

        # --------------------------------------------------
        # 6. Save per-instance CSV
        # --------------------------------------------------
        with csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            fieldnames = [
                "instance",
                "bins_used",
                "runtime_seconds",
            ]

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(instance_results)

        # --------------------------------------------------
        # 7. Log final summary
        # --------------------------------------------------
        logger.info("")
        logger.info("========== DEEPACO RESULTS ==========")
        logger.info("Instances: %s", len(test_list))
        logger.info("Total bins: %s", total_bins)
        logger.info("Average bins: %.3f", average_bins)
        logger.info("Runtime: %.3f seconds", elapsed)
        logger.info("CSV results: %s", csv_path)
        logger.info("Log file: %s", log_path)
        logger.info("=====================================")

    except Exception:
        logger.exception("DeepACO evaluation failed")
        raise

    finally:
        for handler in logger.handlers[:]:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
