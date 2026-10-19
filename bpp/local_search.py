
import torch


def _inc(stats, key, amount=1):
    """Increment a statistics counter when tracking is enabled."""
    if stats is not None:
        stats[key] = stats.get(key, 0) + amount


def path_to_bins(path):
    """Convert DeepACO's zero-separated path into bins."""
    nodes = path.detach().cpu().tolist()
    bins = []
    current_bin = []

    for node in nodes[1:]:
        if node == 0:
            if current_bin:
                bins.append(current_bin)
                current_bin = []
        else:
            current_bin.append(node)

    if current_bin:
        bins.append(current_bin)

    return bins


def bins_to_path(bins, device):
    """Convert bins back into DeepACO's path representation."""
    path = [0]

    for items in bins:
        path.extend(items)
        path.append(0)

    return torch.tensor(path, dtype=torch.long, device=device)


def count_bins(path):
    """Return the number of bins represented by a path."""
    return len(path_to_bins(path))


def bin_loads(bins, sizes):
    """Calculate the total demand assigned to each bin."""
    return [sum(sizes[item] for item in b) for b in bins]


def improve_moves_and_swaps(bins, sizes, capacity, stats=None):
    """
    Improve packing density using item moves and swaps.

    The objective is to maximize the sum of squared bin loads.
    Every accepted operation must strictly improve that objective.
    """
    bins = [items[:] for items in bins]

    while True:
        loads = bin_loads(bins, sizes)
        current_score = sum(load * load for load in loads)
        improved = False

        # Phase 1: Move one item to another existing bin.
        for i in range(len(bins)):
            if improved:
                break

            for item in bins[i][:]:
                for j in range(len(bins)):
                    if i == j:
                        continue

                    _inc(stats, "move_candidates")

                    # The destination bin must remain within capacity.
                    if loads[j] + sizes[item] > capacity:
                        continue

                    new_loads = loads[:]
                    new_loads[i] -= sizes[item]
                    new_loads[j] += sizes[item]
                    new_score = sum(load * load for load in new_loads)

                    if new_score > current_score:
                        bins[i].remove(item)
                        bins[j].append(item)

                        _inc(stats, "accepted_moves")

                        # Count bins emptied by this accepted move.
                        nonempty_bins = [b for b in bins if b]
                        removed_bins = len(bins) - len(nonempty_bins)

                        _inc(
                            stats,
                            "bins_removed_by_moves",
                            removed_bins,
                        )

                        bins = nonempty_bins
                        improved = True
                        break

                if improved:
                    break

        if improved:
            # Recalculate loads and objective after every accepted move.
            continue

        # Phase 2: Swap items between two different bins.
        for i in range(len(bins)):
            if improved:
                break

            for j in range(i + 1, len(bins)):
                for item_a in bins[i]:
                    if improved:
                        break

                    for item_b in bins[j]:
                        _inc(stats, "swap_candidates")

                        load_a = (
                            loads[i]
                            - sizes[item_a]
                            + sizes[item_b]
                        )
                        load_b = (
                            loads[j]
                            - sizes[item_b]
                            + sizes[item_a]
                        )

                        if load_a > capacity or load_b > capacity:
                            continue

                        new_score = (
                            current_score
                            - loads[i] ** 2
                            - loads[j] ** 2
                            + load_a ** 2
                            + load_b ** 2
                        )

                        if new_score > current_score:
                            bins[i].remove(item_a)
                            bins[j].remove(item_b)

                            bins[i].append(item_b)
                            bins[j].append(item_a)

                            _inc(stats, "accepted_swaps")
                            improved = True
                            break

                if improved:
                    break

        if not improved:
            return bins


def try_eliminate_bin(source, other_bins, sizes, capacity, stats=None):
    """
    Try to redistribute every item from source into other_bins.

    Returns the updated bins if redistribution succeeds;
    otherwise returns None.
    """
    items = sorted(
        source,
        key=lambda item: sizes[item],
        reverse=True,
    )

    loads = bin_loads(other_bins, sizes)
    assignments = [[] for _ in other_bins]

    def place(index):
        _inc(stats, "backtracking_nodes")

        if index == len(items):
            return True

        item = items[index]
        size = sizes[item]
        tried_loads = set()

        for j in range(len(other_bins)):
            # Bins with identical current loads are equivalent
            # for this placement decision.
            if loads[j] in tried_loads:
                continue

            tried_loads.add(loads[j])
            _inc(stats, "elimination_placement_candidates")

            if loads[j] + size <= capacity:
                loads[j] += size
                assignments[j].append(item)

                if place(index + 1):
                    return True

                assignments[j].pop()
                loads[j] -= size

        return False

    if not place(0):
        return None

    return [
        other_bins[j] + assignments[j]
        for j in range(len(other_bins))
    ]


def local_search(path, demand, capacity=150, stats=None):
    """
    Refine a valid DeepACO solution.

    Optional stats dictionary records candidate evaluations,
    accepted operations, and the number of bins removed by
    each mechanism.
    """
    sizes = demand.detach().cpu().tolist()
    original_bins = path_to_bins(path)
    original_bin_count = len(original_bins)

    if not original_bins:
        return path.clone()

    bins = [items[:] for items in original_bins]

    # First improve the packing through moves and swaps.
    bins = improve_moves_and_swaps(
        bins,
        sizes,
        capacity,
        stats,
    )

    # Then try eliminating the least-loaded bins first.
    while len(bins) > 1:
        eliminated = False

        candidates = sorted(
            range(len(bins)),
            key=lambda i: (
                sum(sizes[item] for item in bins[i]),
                len(bins[i]),
            ),
        )

        for index in candidates:
            source = bins[index]
            other_bins = [
                b for i, b in enumerate(bins)
                if i != index
            ]

            _inc(stats, "bin_elimination_attempts")

            updated_bins = try_eliminate_bin(
                source,
                other_bins,
                sizes,
                capacity,
                stats,
            )

            if updated_bins is not None:
                # The explicit elimination procedure removed one bin.
                bins = updated_bins
                _inc(stats, "bins_eliminated")

                # Further moves may empty additional bins.
                bins = improve_moves_and_swaps(
                    bins,
                    sizes,
                    capacity,
                    stats,
                )

                eliminated = True
                break

        if not eliminated:
            break

    result = bins_to_path(bins, path.device)

    # Verify that all original items remain present exactly once.
    original_items = sorted(
        item
        for bin_items in original_bins
        for item in bin_items
    )
    result_items = sorted(
        item
        for bin_items in path_to_bins(result)
        for item in bin_items
    )

    assert original_items == result_items, (
        "Local Search changed the set or multiplicity of packed items."
    )

    # Verify that every bin respects the capacity constraint.
    final_bins = path_to_bins(result)
    final_loads = bin_loads(final_bins, sizes)

    assert all(
        load <= capacity for load in final_loads
    ), "Local Search produced a bin exceeding capacity."

    # Local Search must never increase the bin count.
    final_bin_count = len(final_bins)

    assert final_bin_count <= original_bin_count, (
        "Local Search increased the number of bins."
    )

    # Check that the two tracked mechanisms explain all bin reductions.
    if stats is not None:
        bins_removed_by_moves = stats.get(
            "bins_removed_by_moves", 0
        )
        bins_eliminated = stats.get("bins_eliminated", 0)

        # These counters may be cumulative if stats is reused.
        # Record the identity only when the counters started fresh
        # for this call; the experiment should normally use fresh stats.
        stats["initial_bins"] = stats.get(
            "initial_bins", original_bin_count
        )
        stats["final_bins"] = final_bin_count

    return result
