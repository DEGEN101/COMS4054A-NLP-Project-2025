import json
import os
import sys
import matplotlib.pyplot as plt

# === Check input file ===
if len(sys.argv) < 3:
    print("Usage: python plot_compare_models.py <baseline_json> <ce_json>")
    sys.exit(1)

baseline_path = sys.argv[1]
ce_path = sys.argv[2]

# === Validate files ===
for p in [baseline_path, ce_path]:
    if not os.path.exists(p):
        print(f"Error: File '{p}' not found.")
        sys.exit(1)

# === Load Data ===
with open(baseline_path, "r") as f:
    baseline = json.load(f)

with open(ce_path, "r") as f:
    ce = json.load(f)

# === Ensure plots folder exists ===
os.makedirs("plots", exist_ok=True)

# Prefix names
baseline_prefix = "Baseline Transformer"
ce_prefix = "Card Embedding Transformer"

epochs = range(1, len(baseline["train_losses"]) + 1)

def save_fig(fig, name):
    path = os.path.join("plots", f"compare_{name}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# === Helper to make side-by-side plots ===
def plot_metric(metric_name, ylabel, title):
    fig, axes = plt.subplots(1, 2, figsize=(14,5), sharey=True)

    # Baseline
    axes[0].plot(epochs, baseline[f"train_{metric_name}"], label="Train", linewidth=2)
    axes[0].plot(epochs, baseline[f"val_{metric_name}"], label="Validation", linewidth=2)
    axes[0].set_title(f"{baseline_prefix}")
    axes[0].set_ylabel(ylabel)
    axes[0].set_xlabel("Epochs")
    axes[0].grid(True)
    axes[0].legend()

    # CE Transformer
    axes[1].plot(epochs, ce[f"train_{metric_name}"], label="Train", linewidth=2)
    axes[1].plot(epochs, ce[f"val_{metric_name}"], label="Validation", linewidth=2)
    axes[1].set_title(f"{ce_prefix}")
    axes[1].set_xlabel("Epochs")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle(title, fontsize=14)
    save_fig(fig, metric_name)

# === Generate Plots ===
plot_metric("losses",        "Loss",        "Training vs Validation Loss")
plot_metric("accs",          "Accuracy",    "Training vs Validation Accuracy")
plot_metric("perplexities",  "Perplexity",  "Training vs Validation Perplexity")

print("\nAll comparison plots saved in /plots as compare_*.png")
