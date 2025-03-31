import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv("distances_log.csv")

# Add point pair id
df["point_pair_id"] = df.groupby(["num_decoders", "vae_id"]).cumcount()

# Compute mean and std
grouped = df.groupby(["num_decoders", "point_pair_id"])["distance"]
summary = grouped.agg(["mean", "std"]).reset_index()

# Compute CoV
summary["cov"] = summary["std"] / summary["mean"]


os.makedirs("figures/png/cov", exist_ok=True)
os.makedirs("figures/svg/cov", exist_ok=True)

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=summary, x="num_decoders", y="cov")

plt.title("CoV against Num Decoders")
plt.xlabel("Num Decoders")
plt.ylabel("CoV")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/png/cov/cov_plot.png", dpi=300)
plt.savefig("figures/svg/cov/cov_plot.svg")
#plt.show()