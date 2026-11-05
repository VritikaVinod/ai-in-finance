"""Create concise, presentation-ready charts from the saved evaluation outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {"Logistic Regression": "#0E7C66", "Random Forest": "#C2410C", "Extra Trees": "#1E3A5F"}
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

metrics = pd.read_csv(APP / "model_comparison_metrics.csv")
metric_cols = ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc_average_precision"]
metric_labels = ["Accuracy", "Balanced accuracy", "ROC-AUC", "PR-AUC"]

fig, ax = plt.subplots(figsize=(10, 5.5))
width = 0.24
x = range(len(metric_cols))
for i, row in metrics.iterrows():
    vals = [row[c] for c in metric_cols]
    ax.bar([v + (i - 1) * width for v in x], vals, width, label=row["model"], color=COLORS[row["model"]])
ax.set_xticks(list(x), metric_labels)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
ax.set_title("Out-of-fold model comparison at a fixed 0.50 threshold", loc="left", weight="bold")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
ax.grid(axis="y", color="#E7E1D6", linewidth=0.8)
fig.tight_layout()
fig.savefig(OUT / "model_performance.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.8))
for _, row in metrics.iterrows():
    ax.bar(row["model"], row["recall_failure"], color=COLORS[row["model"]])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Recall of failed startups")
ax.set_title("Failure recall is limited by the small failure class", loc="left", weight="bold")
ax.grid(axis="y", color="#E7E1D6", linewidth=0.8)
fig.tight_layout()
fig.savefig(OUT / "failure_recall.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
labels = ["Operating", "Failed"]
values = [161, 8]
bars = ax.bar(labels, values, color=["#0E7C66", "#B42318"], width=0.55)
ax.set_ylabel("Companies")
ax.set_title("Evaluation cohort: 169 companies", loc="left", weight="bold")
ax.grid(axis="y", color="#E7E1D6", linewidth=0.8)
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 2, str(value), ha="center", weight="bold")
fig.tight_layout()
fig.savefig(OUT / "cohort_balance.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Wrote charts to {OUT}")
