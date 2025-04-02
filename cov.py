import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_and_prepare_data(csv_path, distance_type):
    df = pd.read_csv(csv_path)
    df["point_pair_id"] = df.groupby(["num_decoders", "vae_id"]).cumcount()

    distance_col = "distance" if distance_type == "geodesic" else "euclidean_distance"

    grouped = df.groupby(["num_decoders", "point_pair_id"])[distance_col]
    summary = grouped.agg(["mean", "std"]).reset_index()
    summary["cov"] = summary["std"] / summary["mean"]
    summary["distance_type"] = distance_type.capitalize()

    return summary

def plot_cov_boxplot(geo_summary, eucl_summary, file_name):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    sns.boxplot(data=geo_summary, x="num_decoders", y="cov", ax=axes[0])
    axes[0].set_title("Geodesic CoV vs Number of Decoders (Boxplot)")

    sns.boxplot(data=eucl_summary, x="num_decoders", y="cov", ax=axes[1])
    axes[1].set_title("Euclidean CoV vs Number of Decoders (Boxplot)")

    for ax in axes:
        ax.set_ylabel("Coefficient of Variation")
        ax.set_xlabel("Number of Decoders")

    plt.tight_layout()
    plt.savefig(f"figures/png/cov/{file_name}_box.png", dpi=300)
    plt.savefig(f"figures/svg/cov/{file_name}_box.svg")
    plt.close()

def plot_cov_lineplot(geo_summary, eucl_summary, file_name):
    geo_avg = geo_summary.groupby("num_decoders")["cov"].mean().reset_index()
    eucl_avg = eucl_summary.groupby("num_decoders")["cov"].mean().reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(geo_avg["num_decoders"], geo_avg["cov"], label="Geodesic")
    plt.plot(eucl_avg["num_decoders"], eucl_avg["cov"], label="Euclidean")

    plt.title("Mean CoV vs Number of Decoders (Line Plot)")
    plt.xlabel("Number of Decoders")
    plt.ylabel("Mean Coefficient of Variation")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"figures/png/cov/{file_name}_line.png", dpi=300)
    plt.savefig(f"figures/svg/cov/{file_name}_line.svg")
    plt.close()

def generate_plots(csv_path, file_name):
    os.makedirs("figures/png/cov", exist_ok=True)
    os.makedirs("figures/svg/cov", exist_ok=True)

    geo_summary = load_and_prepare_data(csv_path, "geodesic")
    eucl_summary = load_and_prepare_data(csv_path, "euclidean")

    plot_cov_boxplot(geo_summary, eucl_summary, file_name)
    plot_cov_lineplot(geo_summary, eucl_summary, file_name)

if __name__ == "__main__":
    generate_plots("10distances_log_500ep.csv", "cov_plot_combined_10_500ep")
    generate_plots("25distances_log_500ep.csv", "cov_plot_combined_25_500ep")