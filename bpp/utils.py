import os
import torch
from torch_geometric.data import Data

# Emanuel Falkenauer. A hybrid grouping genetic algorithm
# for bin packing. Journal of Heuristics, 2:5–30, 1996.

DEMAND_LOW = 20
DEMAND_HIGH = 100


def gen_instance(n, device):
    """
    Generate one bin-packing instance.

    Returns a tensor of shape (n + 1,).
    The first element is a dummy zero-demand item.
    """
    demands = torch.randint(
        low=DEMAND_LOW,
        high=DEMAND_HIGH + 1,
        size=(n,),
        device=device
    )

    all_demands = torch.cat((
        torch.zeros((1,), device=device, dtype=demands.dtype),
        demands
    ))

    return all_demands


def gen_pyg_data(demands, device="cpu"):
    """
    Convert item demands into a PyTorch Geometric graph.
    """
    demands = demands.to(device)
    n = demands.size(0)

    nodes = torch.arange(n, device=device)
    u = nodes.repeat(n)
    v = torch.repeat_interleave(nodes, n)

    edge_index = torch.stack((u, v))
    edge_attr = torch.ones(
        (edge_index.size(1), 1),
        device=device
    )

    x = demands

    pyg_data = Data(
        x=x.unsqueeze(1),
        edge_attr=edge_attr,
        edge_index=edge_index
    )

    return pyg_data


def load_test_dataset(problem_size, device="cpu"):
    """
    Load a previously generated test dataset safely.

    The dataset generated below is a plain tensor, so
    weights_only=True is sufficient.
    """
    dataset_path = os.path.join(
        "..",
        "data",
        "bpp",
        f"testDataset-{problem_size}.pt"
    )

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"Test dataset not found: {os.path.abspath(dataset_path)}\n"
            "Generate it by running: python utils.py"
        )

    dataset = torch.load(
        dataset_path,
        map_location=device,
        weights_only=True
    )

    return dataset


if __name__ == "__main__":
    import pathlib

    data_dir = pathlib.Path("../data/bpp")
    data_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(123456)

    for n in [120]:
        inst_list = []

        for _ in range(100):
            demands = gen_instance(n, "cpu")
            inst_list.append(demands)

        test_dataset = torch.stack(inst_list)

        output_path = data_dir / f"testDataset-{n}.pt"
        torch.save(test_dataset, output_path)

        print(f"Dataset saved to: {output_path.resolve()}")
        print(f"Dataset shape: {tuple(test_dataset.shape)}")