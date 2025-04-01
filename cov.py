import pandas as pd
import numpy as np
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

def plot_combined_cov(csv_path="distances_log.csv", file_name="cov_plot_combined", plot_type="box"):
    geo_summary = load_and_prepare_data(csv_path, "geodesic")
    eucl_summary = load_and_prepare_data(csv_path, "euclidean")

    os.makedirs("figures/png/cov", exist_ok=True)
    os.makedirs("figures/svg/cov", exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    if plot_type == "box":
        sns.boxplot(data=geo_summary, x="num_decoders", y="cov", ax=axes[0])
        axes[0].set_title("Geodesic CoV vs Num Decoders (Boxplot)")

        sns.boxplot(data=eucl_summary, x="num_decoders", y="cov", ax=axes[1])
        axes[1].set_title("Euclidean CoV vs Num Decoders (Boxplot)")

    elif plot_type == "line":
        geo_avg = geo_summary.groupby("num_decoders")["cov"].mean().reset_index()
        eucl_avg = eucl_summary.groupby("num_decoders")["cov"].mean().reset_index()

        axes[0].plot(geo_avg["num_decoders"], geo_avg["cov"], marker='o')
        axes[0].set_title("Geodesic CoV vs Num Decoders (Mean Line Plot)")

        axes[1].plot(eucl_avg["num_decoders"], eucl_avg["cov"], marker='o')
        axes[1].set_title("Euclidean CoV vs Num Decoders (Mean Line Plot)")

    else:
        raise ValueError("Invalid plot_type. Use 'box' or 'line'.")

    for ax in axes:
        ax.set_ylabel("CoV")
        ax.set_xlabel("Num Decoders")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"figures/png/cov/{file_name}_{plot_type}.png", dpi=300)
    plt.savefig(f"figures/svg/cov/{file_name}_{plot_type}.svg")
    # plt.show()

if __name__ == "__main__":
    plot_combined_cov("10distances_log.csv", "cov_plot_combined_10", plot_type="box")
    plot_combined_cov("10distances_log.csv", "cov_plot_combined_10", plot_type="line")
    plot_combined_cov("25distances_log.csv", "cov_plot_combined_25", plot_type="box")
    plot_combined_cov("25distances_log.csv", "cov_plot_combined_25", plot_type="line")
