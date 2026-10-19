
CAPACITY = 150


def best_fit_decreasing(demands, capacity=CAPACITY):
    """Pack items using Best-Fit Decreasing."""

    if hasattr(demands, "detach"):
        demands = demands.detach().cpu().tolist()

    # Index 0 is the dummy node in DeepACO.
    items = sorted(demands[1:], reverse=True)

    bins = []
    bin_loads = []

    for item in items:
        best_bin = -1
        smallest_remaining = float("inf")

        # Find the fullest bin that can still fit this item.
        for i, load in enumerate(bin_loads):
            remaining = capacity - load

            if item <= remaining and remaining - item < smallest_remaining:
                smallest_remaining = remaining - item
                best_bin = i

        if best_bin == -1:
            bins.append([item])
            bin_loads.append(item)
        else:
            bins[best_bin].append(item)
            bin_loads[best_bin] += item

    return bins
