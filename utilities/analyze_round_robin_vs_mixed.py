import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# =====================
# --- Configuration ---
# =====================
SEEDS = [31, 42, 45, 69, 420]
PLOTS_DIR = "./plots"
RESULTS_DIR = "./results/training_methods"
MIXED_PREFIX = "baseline_transformer_performace_seed-"
RR_PREFIX = "baseline_transformer_round_robin_performace_seed-"

CONTEXTS = ["colour", "shape", "quantity"]
SAVE_FIGS = True

def load_results(prefix):
    results = []
    for seed in SEEDS:
        path = os.path.join(RESULTS_DIR, f"{prefix}{seed}.json")
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue
        with open(path) as f:
            results.append(json.load(f))
    return results


def mean_std(arr):
    arr = np.array(arr)
    return np.mean(arr), np.std(arr)


def extract_final(results, key):
    """Extract final value of a metric list (e.g., val_accs)."""
    vals = []
    for r in results:
        if key in r and len(r[key]) > 0:
            vals.append(r[key][-1])
    return vals


def extract_context(results, context, metric):
    """Safely extract per-context metric if available."""
    vals = []
    for r in results:
        # Handle new-style 'val_context' format
        if "val_context" in r:
            try:
                ctx_vals = [epoch[context][metric] for epoch in r["val_context"] if context in epoch]
                if ctx_vals:
                    vals.append(ctx_vals[-1])  # last epoch
            except Exception:
                continue
        # Handle older 'per_context' format (list of dicts)
        elif "per_context" in r and context in r["per_context"]:
            try:
                vals.append(r["per_context"][context][metric][-1])
            except Exception:
                continue
    return vals


def paired_test(rr, mixed):
    rr, mixed = np.array(rr), np.array(mixed)
    t, p = ttest_rel(rr, mixed)
    diff = rr - mixed
    cohen_d = diff.mean() / diff.std(ddof=1)
    return diff.mean(), diff.std(ddof=1), t, p, cohen_d


def plot_curve(mixed_results, rr_results, key="val_accs", title="Validation Accuracy vs Epoch"):
    arr_m = np.array([r[key] for r in mixed_results if key in r])
    arr_rr = np.array([r[key] for r in rr_results if key in r])

    # skip if missing data
    if len(arr_m) == 0 or len(arr_rr) == 0:
        print(f"Skipping curve plot ({key}) — missing data")
        return

    mean_m, std_m = arr_m.mean(0), arr_m.std(0)
    mean_rr, std_rr = arr_rr.mean(0), arr_rr.std(0)
    epochs = np.arange(1, len(mean_m) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, mean_m, label="Mixed", lw=2)
    plt.fill_between(epochs, mean_m - std_m, mean_m + std_m, alpha=0.2)
    plt.plot(epochs, mean_rr, label="Round-Robin", lw=2)
    plt.fill_between(epochs, mean_rr - std_rr, mean_rr + std_rr, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(os.path.join(PLOTS_DIR, "val_accuracy_comparison.png"), dpi=200)
    plt.show()


# =====================
# --- Load Results ---
# =====================
print("Loading results...")
mixed = load_results(MIXED_PREFIX)
rr = load_results(RR_PREFIX)

if not mixed or not rr:
    raise RuntimeError("Could not find required result files. Check paths and prefixes.")

# =====================
# --- Overall Validation Accuracy ---
# =====================
val_acc_mixed = extract_final(mixed, "val_accs")
val_acc_rr = extract_final(rr, "val_accs")

mean_m, std_m = mean_std(val_acc_mixed)
mean_rr, std_rr = mean_std(val_acc_rr)
diff_mean, diff_std, t, p, d = paired_test(val_acc_rr, val_acc_mixed)

print("\n--- Overall Validation Accuracy Comparison ---")
print(f"Mixed        : {mean_m:.4f} ± {std_m:.4f}")
print(f"Round-Robin  : {mean_rr:.4f} ± {std_rr:.4f}")
print(f"Δ (RR-Mixed) : {diff_mean:+.4f} ± {diff_std:.4f} | t={t:.3f}, p={p:.4f}, d={d:.3f}")

# =====================
# --- Test Accuracy ---
# =====================
test_acc_mixed = [r["test_acc"] for r in mixed if "test_acc" in r]
test_acc_rr = [r["test_acc"] for r in rr if "test_acc" in r]

if test_acc_mixed and test_acc_rr:
    mean_m, std_m = mean_std(test_acc_mixed)
    mean_rr, std_rr = mean_std(test_acc_rr)
    diff_mean, diff_std, t, p, d = paired_test(test_acc_rr, test_acc_mixed)
    print("\n--- Test Accuracy Comparison ---")
    print(f"Mixed        : {mean_m:.4f} ± {std_m:.4f}")
    print(f"Round-Robin  : {mean_rr:.4f} ± {std_rr:.4f}")
    print(f"Δ (RR-Mixed) : {diff_mean:+.4f} ± {diff_std:.4f} | t={t:.3f}, p={p:.4f}, d={d:.3f}")
else:
    print("\nTest accuracy not found in results files.")

# =====================
# --- Per Context Accuracy ---
# =====================
if any("val_context" in r or "per_context" in r for r in rr + mixed):
    print("\n--- Per-Context Validation Accuracy ---")
    print(f"{'Context':<12} {'Mixed (mean±std)':<12} \t{'RR (mean±std)'} \t{'Δ±std':<14} {'p':<8} {'d'}")
    print("-" * 80)
    for ctx in CONTEXTS:
        m_vals = extract_context(mixed, ctx, "acc")
        rr_vals = extract_context(rr, ctx, "acc")
        if not m_vals or not rr_vals:
            continue
        m_mean, m_std = mean_std(m_vals)
        rr_mean, rr_std = mean_std(rr_vals)
        diff_mean, diff_std, t, p, d = paired_test(rr_vals, m_vals)
        print(f"{ctx:<12} {m_mean:.3f}±{m_std:.3f}   \t{rr_mean:.3f}±{rr_std:.3f}   {diff_mean:+.3f}±{diff_std:.3f}   {p:.4f}   {d:.3f}")
else:
    print("\nSkipping per-context analysis (not found in any results files).")

# =====================
# --- Plots ---
# =====================
plot_curve(mixed, rr, "val_accs", "Validation Accuracy Across Seeds")

# --- Optional: Per-context bar plot ---
means_m, means_rr, stds_m, stds_rr = [], [], [], []
for ctx in CONTEXTS:
    m_vals = extract_context(mixed, ctx, "acc")
    rr_vals = extract_context(rr, ctx, "acc")
    if not m_vals or not rr_vals:
        continue
    m_mean, m_std = mean_std(m_vals)
    rr_mean, rr_std = mean_std(rr_vals)
    means_m.append(m_mean)
    stds_m.append(m_std)
    means_rr.append(rr_mean)
    stds_rr.append(rr_std)

if means_m:
    x = np.arange(len(means_m))
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, means_m, width, yerr=stds_m, label='Mixed', capsize=4)
    plt.bar(x + width / 2, means_rr, width, yerr=stds_rr, label='Round-Robin', capsize=4)
    plt.xticks(x, CONTEXTS)
    plt.ylabel("Validation Accuracy")
    plt.title("Per-Context Validation Accuracy (mean ± std)")
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(os.path.join(PLOTS_DIR, "per_context_comparison.png"), dpi=200)
    plt.show()
