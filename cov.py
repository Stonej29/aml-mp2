import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_and_prepare_data(csv_path, distance_type):
    df = pd.read_csv(csv_path)
    df["point_pair_id"] = df.groupby(["num_decoders", "vae_id"]).cumcount()

    # Use the appropriate distance column
    distance_col = "distance" if distance_type == "geodesic" else "euclidean_distance"
    
    # Group and compute mean/std/CoV
    grouped = df.groupby(["num_decoders", "point_pair_id"])[distance_col]
    summary = grouped.agg(["mean", "std"]).reset_index()
    summary["cov"] = summary["std"] / summary["mean"]
    summary["distance_type"] = distance_type.capitalize()
    
    return summary

def plot_combined_cov(csv_path="distances_log.csv"):
    # Prepare both summaries
    geo_summary = load_and_prepare_data(csv_path, "geodesic")
    eucl_summary = load_and_prepare_data(csv_path, "euclidean")

    # Create output dirs
    os.makedirs("figures/png/cov", exist_ok=True)
    os.makedirs("figures/svg/cov", exist_ok=True)

    # Plot both in one figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    sns.boxplot(data=geo_summary, x="num_decoders", y="cov", ax=axes[0])
    axes[0].set_title("Geodesic CoV vs Num Decoders")
    axes[0].set_ylabel("CoV")
    axes[0].grid(True)

    sns.boxplot(data=eucl_summary, x="num_decoders", y="cov", ax=axes[1])
    axes[1].set_title("Euclidean CoV vs Num Decoders")
    axes[1].set_xlabel("Num Decoders")
    axes[1].set_ylabel("CoV")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("figures/png/cov/cov_plot_combined.png", dpi=300)
    plt.savefig("figures/svg/cov/cov_plot_combined.svg")
    # plt.show()

if __name__ == "__main__":
    plot_combined_cov()