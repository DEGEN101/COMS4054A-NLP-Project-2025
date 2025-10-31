import json
import os
import matplotlib.pyplot as plt
import numpy as np

# =========================================
# Config
# =========================================
N_values = [4, 5, 6, 7]
base_dir = "results/scaling_laws"
file_template = "ce_transformer_performace_N-{}.json"

test_acc_dict = {}
test_perplexity_dict = {}

os.makedirs("plots", exist_ok=True)

# =========================================
# Load Test Metrics from Logs
# =========================================
for N in N_values:
    file_path = os.path.join(base_dir, file_template.format(N))

    if not os.path.exists(file_path):
        print(f"[WARN] Missing file: {file_path}")
        continue

    with open(file_path, "r") as f:
        data = json.load(f)

    test_acc_dict[N] = data["test_acc"]
    test_perplexity_dict[N] = data["test_perplexity"]

# Filter valid Ns (in case some logs missing)
valid_N = sorted(test_acc_dict.keys())
acc_values = [test_acc_dict[n] for n in valid_N]
ppl_values = [test_perplexity_dict[n] for n in valid_N]

# =========================================
# Plot Side-by-Side
# =========================================
plt.figure(figsize=(12, 5))

# ---------------- Accuracy ----------------
plt.subplot(1, 2, 1)
plt.plot(valid_N, acc_values, marker="o", linewidth=3, markersize=8)
plt.title("Test Accuracy vs N", fontsize=14)
plt.xlabel("N (Colours / Shapes / Quantities)", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(valid_N, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(alpha=0.3, linestyle="--")
plt.ylim(0, 1)
plt.axhline(0.5, linestyle="--", alpha=0.2)  # reference chance-ish baseline
plt.box(True)

# ---------------- Perplexity ----------------
plt.subplot(1, 2, 2)
plt.plot(valid_N, ppl_values, marker="o", linewidth=3, markersize=8)
plt.title("Test Perplexity vs N", fontsize=14)
plt.xlabel("N (Colours / Shapes / Quantities)", fontsize=12)
plt.ylabel("Perplexity", fontsize=12)
plt.xticks(valid_N, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(alpha=0.3, linestyle="--")
plt.axhline(1.0, linestyle="--", alpha=0.2)  # reference baseline
plt.box(True)

plt.tight_layout()

plt.savefig(os.path.join("plots", "wcst_scaling_accuracy_perplexity_vs_N.png"))
plt.show()

# Print values for logging
print("\n=== Test Accuracy & Perplexity by N ===")
for n in valid_N:
    print(f"N={n}: ACC={test_acc_dict[n]:.4f}, PPL={test_perplexity_dict[n]:.4f}")
