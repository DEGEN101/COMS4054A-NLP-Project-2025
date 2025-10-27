import json
import os
import sys
import matplotlib.pyplot as plt

# === Check input file ===
if len(sys.argv) < 2:
    print("Usage: python plot_performace_metrics.py <path_to_json_file>")
    sys.exit(1)

json_path = sys.argv[1]

# === Validate file ===
if not os.path.exists(json_path):
    print(f"Error: File '{json_path}' not found.")
    sys.exit(1)

# === Load Data ===
with open(json_path, "r") as f:
    data = json.load(f)

# === Ensure plots folder exists ===
os.makedirs("plots", exist_ok=True)

# === Extract prefix from filename ===
filename = os.path.basename(json_path)
name_without_ext = os.path.splitext(filename)[0]
prefix_words = name_without_ext.split("_")[:2]  # first two words
prefix = "_".join(prefix_words)

# === Epochs ===
epochs = range(1, len(data["train_losses"]) + 1)

# === Helper function to save plots ===
def save_plot(fig, name):
    path = os.path.join("plots", f"{prefix}_{name}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# === Plot 1: Loss ===
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, data["train_losses"], label="Train Loss", linewidth=2)
ax.plot(epochs, data["val_losses"], label="Validation Loss", linewidth=2)
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Training vs Validation Loss")
ax.legend()
ax.grid(True)
save_plot(fig, "loss_curve")

# === Plot 2: Accuracy ===
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, data["train_accs"], label="Train Accuracy", linewidth=2)
ax.plot(epochs, data["val_accs"], label="Validation Accuracy", linewidth=2)
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")
ax.set_title("Training vs Validation Accuracy")
ax.legend()
ax.grid(True)
save_plot(fig, "accuracy_curve")

# === Plot 3: Perplexity ===
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, data["train_perplexities"], label="Train Perplexity", linewidth=2)
ax.plot(epochs, data["val_perplexities"], label="Validation Perplexity", linewidth=2)
ax.set_xlabel("Epochs")
ax.set_ylabel("Perplexity")
ax.set_title("Training vs Validation Perplexity")
ax.legend()
ax.grid(True)
save_plot(fig, "perplexity_curve")

print(f"\nAll plots saved successfully in the 'plots' folder using prefix '{prefix}_'.")
