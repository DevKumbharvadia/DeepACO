import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"logs\final_scalability_analysis"
os.makedirs(OUT, exist_ok=True)

SUMMARY_FILES = [
    r"logs\scalability_20261017_220218\scalability_summary.csv",
    r"logs\scalability_20261017_221631\scalability_summary.csv",
]
INSTANCE_FILES = [
    r"logs\scalability_20261017_220218\scalability_per_instance.csv",
    r"logs\scalability_20261017_221631\scalability_per_instance.csv",
]
METHODS = ["FF", "FFD", "BFD", "DeepACO", "DeepACO+LS"]


def load_files(paths):
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError("Required results file not found: " + path)
    return pd.concat([pd.read_csv(path) for path in paths],
                     ignore_index=True)


# 1. Combine summary data from all seven sizes.
summary = load_files(SUMMARY_FILES)
summary["size"] = pd.to_numeric(summary["size"])

for col in ["average_bins", "average_unused_capacity",
            "average_runtime_seconds"]:
    summary[col] = pd.to_numeric(summary[col])

if summary.duplicated(["size", "method"]).any():
    raise ValueError("Duplicate size/method summary rows detected.")

summary = summary.sort_values(["size", "method"])
summary.to_csv(os.path.join(OUT, "scalability_combined_7_sizes.csv"),
               index=False)


# 2. Generate the three main comparison graphs.
def plot_metric(column, ylabel, title, filename, log_y=False):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for method in METHODS:
        data = summary[summary["method"] == method].sort_values("size")
        ax.plot(data["size"], data[column], marker="o",
                linewidth=2, label=method)

    ax.set_xlabel("Number of items")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(summary["size"].unique()))

    if log_y:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, filename), dpi=300)
    plt.close(fig)


plot_metric("average_bins", "Average number of bins",
            "Packing Quality vs. Problem Size", "average_bins.png")

plot_metric("average_unused_capacity", "Average unused capacity",
            "Unused Capacity vs. Problem Size",
            "average_unused_capacity.png")

plot_metric("average_runtime_seconds", "Average runtime (seconds)",
            "Runtime vs. Problem Size", "average_runtime.png",
            log_y=True)


# 3. Compare DeepACO and DeepACO+LS on paired instances.
instances = load_files(INSTANCE_FILES)

required = {"size", "instance", "method", "bins", "valid"}
missing = required - set(instances.columns)
if missing:
    raise ValueError("Missing per-instance columns: " + str(missing))

if instances["valid"].astype(str).str.lower().isin(
        ["false", "0"]).any():
    raise ValueError("Invalid solutions found in experiment results.")

pair = instances[
    instances["method"].isin(["DeepACO", "DeepACO+LS"])
].copy()

if pair.duplicated(["size", "instance", "method"]).any():
    raise ValueError("Duplicate paired instance records detected.")

wide = pair.pivot(
    index=["size", "instance"],
    columns="method",
    values="bins"
).reset_index()

if wide[["DeepACO", "DeepACO+LS"]].isna().any().any():
    raise ValueError("Some instances are missing a paired result.")

wide["bins_saved"] = wide["DeepACO"] - wide["DeepACO+LS"]
wide["outcome"] = wide["bins_saved"].map(
    lambda x: "Improved" if x > 0 else
    "Worsened" if x < 0 else "Unchanged"
)

wide.to_csv(os.path.join(OUT, "paired_local_search_per_instance.csv"),
            index=False)

paired_summary = wide.groupby("size").agg(
    instances=("bins_saved", "size"),
    deepaco_avg_bins=("DeepACO", "mean"),
    deepaco_ls_avg_bins=("DeepACO+LS", "mean"),
    total_bins_saved=("bins_saved", "sum"),
    average_bins_saved=("bins_saved", "mean"),
    improved_instances=("bins_saved", lambda x: int((x > 0).sum())),
    unchanged_instances=("bins_saved", lambda x: int((x == 0).sum())),
    worsened_instances=("bins_saved", lambda x: int((x < 0).sum()))
).reset_index()

paired_summary["relative_improvement_percent"] = (
    paired_summary["average_bins_saved"]
    / paired_summary["deepaco_avg_bins"] * 100
)

paired_summary.to_csv(
    os.path.join(OUT, "paired_local_search_summary.csv"), index=False
)


# 4. Generate the focused local-search graph.
fig, ax = plt.subplots(figsize=(9, 5.5))

for method in ["DeepACO", "DeepACO+LS"]:
    data = summary[summary["method"] == method].sort_values("size")
    ax.plot(data["size"], data["average_bins"], marker="o",
            linewidth=2, label=method)

ax.set_xlabel("Number of items")
ax.set_ylabel("Average number of bins")
ax.set_title("Effect of Local Search on DeepACO")
ax.set_xticks(sorted(summary["size"].unique()))
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "deepaco_vs_local_search.png"), dpi=300)
plt.close(fig)


# 5. Print validation and results.
print("Analysis completed.")
print("Summary rows:", len(summary))
print("Paired instances:", len(wide))
print("Improved:", int((wide["bins_saved"] > 0).sum()))
print("Unchanged:", int((wide["bins_saved"] == 0).sum()))
print("Worsened:", int((wide["bins_saved"] < 0).sum()))
print("Total bins saved:", int(wide["bins_saved"].sum()))
print("\nPaired results by size:")
print(paired_summary.round(3).to_string(index=False))
print("\nOutput directory:", OUT)
