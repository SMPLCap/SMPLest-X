"""

"""
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

LOG_PATHS = ["outputs/train_worldpose_ft_20260613_153309/log/train_logs.txt"]

# LOG_PATHS = ["outputs/saved_logs/train_worldpose_ft_20260603_003131/log/train_logs.txt",
#     "outputs/saved_logs/train_worldpose_ft_20260603_202653/log/train_logs.txt",
#     "outputs/saved_logs/train_worldpose_ft_20260605_105411/log/train_logs.txt",
#     "outputs/saved_logs/train_worldpose_ft_20260605_223355/log/train_logs.txt",
#     "outputs/saved_logs/train_worldpose_ft_20260606_171757/log/train_logs.txt",
#     "outputs/saved_logs/train_worldpose_ft_20260606_225445_30_epochs/log/train_logs.txt"]

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

LOSS_KEYS = [
    "loss_smplx_orient",
    "loss_smplx_pose",
    "loss_smplx_shape",
    "loss_smplx_expr",
    "loss_joint_cam",
    "loss_smplx_joint_cam",
    "loss_joint_proj",
    "loss_hand_root",
    "loss_hand_root_chain",
    "loss_total",
]

ITR_RE = re.compile(r"Epoch (\d+)/\d+ itr (\d+)/(\d+):")
LOSS_RE = re.compile(r"(loss_\w+): ([0-9.e+\-]+)")


def parse_log(paths):
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = ANSI_ESCAPE.sub("", line)
                m = ITR_RE.search(line)
                if not m:
                    continue
                epoch, itr, total_itr = int(m.group(1)), int(m.group(2)), int(m.group(3))
                losses = {k: float(v) for k, v in LOSS_RE.findall(line)}
                records.append({"epoch": epoch, "itr": itr, "total_itr": total_itr, **losses})
    return records


def smooth(values, window=50):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_losses(records, smooth_window=50, output_path="training_loss.png"):
    # Process records in file order to handle combined logs with multiple runs
    # (different total_itr per run, or epoch numbers that restart mid-file).
    cum_itr = []
    all_losses = defaultdict(list)
    epoch_boundaries_itr = []   # x-value at first logged step of each epoch (for axvline)
    epoch_boundaries_idx = []   # list index into all_losses (for slicing per-epoch means)
    epoch_labels = []           # epoch number label for each boundary

    prev_ep = None
    prev_total_itr = None
    run_itr_offset = 0          # cumulative itr added at each run boundary

    for r in records:
        ep, itr, total_itr = r["epoch"], r["itr"], r["total_itr"]

        # Detect a new run: epoch went backwards or total_itr changed
        if prev_ep is not None and (ep < prev_ep or total_itr != prev_total_itr):
            run_itr_offset += (prev_ep + 1) * prev_total_itr

        global_itr = run_itr_offset + ep * total_itr + itr

        if ep != prev_ep or total_itr != prev_total_itr:
            epoch_boundaries_itr.append(global_itr)
            epoch_boundaries_idx.append(len(cum_itr))
            epoch_labels.append(ep)

        cum_itr.append(global_itr)
        for k in LOSS_KEYS:
            all_losses[k].append(r.get(k, float("nan")))

        prev_ep = ep
        prev_total_itr = total_itr

    epochs = epoch_labels

    # --- Figure 1: loss_total with smoothing + per-epoch mean ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    fig.suptitle("Training Loss — train_worldpose_ft_20260430_142108", fontsize=13)

    ax = axes[0]
    raw = np.array(all_losses["loss_total"])
    ax.plot(cum_itr, raw, alpha=0.25, color="steelblue", linewidth=0.6, label="raw")
    if smooth_window > 1:
        s = smooth(raw, smooth_window)
        pad = len(raw) - len(s)
        ax.plot(cum_itr[pad:], s, color="steelblue", linewidth=1.5,
                label=f"smooth (w={smooth_window})")
    for i, (ep, bnd) in enumerate(zip(epochs, epoch_boundaries_itr)):
        ax.axvline(bnd, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(bnd + 5, ax.get_ylim()[1] * 0.98 if i == 0 else ax.get_ylim()[1] * 0.98,
                f"Ep{ep}", fontsize=7, color="gray", va="top")
    ax.set_ylabel("loss_total")
    ax.set_xlabel("global iteration")
    ax.legend(fontsize=9)
    ax.set_title("Total Loss (all iterations)")

    # Per-epoch mean bar chart
    ax2 = axes[1]
    epoch_means = [np.nanmean(all_losses["loss_total"][epoch_boundaries_idx[i]:
                   (epoch_boundaries_idx[i+1] if i+1 < len(epoch_boundaries_idx) else None)])
                   for i in range(len(epochs))]
    bars = ax2.bar(epochs, epoch_means, color="steelblue", alpha=0.7, edgecolor="navy")
    for bar, val in zip(bars, epoch_means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean loss_total")
    ax2.set_title("Per-Epoch Mean Total Loss")
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()

    # --- Figure 2: fixed layout ---
    # Row 0: smplx_orient | smplx_pose
    # Row 1: joint_cam    | joint_proj
    layout = [
        ["loss_smplx_orient", "loss_smplx_pose"],
        ["loss_joint_cam",    "loss_joint_proj"],
    ]

    x_pos = list(range(len(epochs)))
    x_labels = [str(ep) for ep in epochs]

    fig2, axs = plt.subplots(2, 2, figsize=(12, 7))
    fig2.suptitle("Loss Components — Per-Epoch Mean", fontsize=13)

    for row, pair in enumerate(layout):
        for col, key in enumerate(pair):
            ax = axs[row][col]
            if key is None:
                ax.set_visible(False)
                continue
            means = [np.nanmean(all_losses[key][epoch_boundaries_idx[i]:
                     (epoch_boundaries_idx[i+1] if i+1 < len(epoch_boundaries_idx) else None)])
                     for i in range(len(epochs))]
            is_total = key == "loss_total"
            ax.plot(x_pos, means, marker="o",
                    linewidth=2.0 if is_total else 1.5,
                    color="navy" if is_total else None)
            ax.set_title(key.replace("loss_", ""), fontsize=10,
                         fontweight="bold" if is_total else "normal")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_xlabel("Epoch")

    plt.tight_layout()
    out2 = output_path.replace(".png", "_components.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.show()


def compute_epoch_means(all_losses, epoch_boundaries_idx, epochs):
    means = {}
    for key in LOSS_KEYS:
        means[key] = [
            np.nanmean(all_losses[key][epoch_boundaries_idx[i]:
                (epoch_boundaries_idx[i + 1] if i + 1 < len(epoch_boundaries_idx) else None)])
            for i in range(len(epochs))
        ]
    return means


def build_epoch_data(records):
    """Return (epochs, epoch_boundaries_idx, all_losses) processing records in file order."""
    all_losses = defaultdict(list)
    epoch_boundaries_idx = []
    epoch_labels = []
    prev_ep = None
    prev_total_itr = None

    for r in records:
        ep, total_itr = r["epoch"], r["total_itr"]
        if ep != prev_ep or total_itr != prev_total_itr:
            epoch_boundaries_idx.append(len(all_losses["loss_total"]))
            epoch_labels.append(ep)
        for k in LOSS_KEYS:
            all_losses[k].append(r.get(k, float("nan")))
        prev_ep = ep
        prev_total_itr = total_itr

    return epoch_labels, epoch_boundaries_idx, all_losses


def plot_paper_total(records, smooth_window=100, output_path="paper_loss_total.pdf"):
    """Clean single-panel total loss curve for the paper."""
    epochs, epoch_boundaries_idx, all_losses = build_epoch_data(records)
    means = compute_epoch_means(all_losses, epoch_boundaries_idx, epochs)

    x = list(range(len(epochs)))
    y = means["loss_total"]

    fig, ax = plt.subplots(figsize=(5, 3))

    raw = np.array(y)
    if smooth_window > 1 and len(raw) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        y_plot = np.convolve(raw, kernel, mode="same")
    else:
        y_plot = raw

    ax.plot(x, y_plot, color="#2166ac", linewidth=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in epochs], fontsize=9)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


def plot_paper_components(records, output_path="paper_loss_components.pdf"):
    """Clean 3x2 component grid for the supplementary."""
    epochs, epoch_boundaries_idx, all_losses = build_epoch_data(records)
    means = compute_epoch_means(all_losses, epoch_boundaries_idx, epochs)

    layout = [
        ["loss_smplx_orient", "loss_smplx_pose"],
        ["loss_joint_cam",    "loss_joint_proj"],
    ]
    titles = {
        "loss_smplx_orient": "Orient.",
        "loss_smplx_pose":   "Pose",
        "loss_joint_cam":    "Joint Cam",
        "loss_joint_proj":   "Joint Proj.",
    }
    colors = {
        "loss_smplx_orient": "#d6604d",
        "loss_smplx_pose":   "#f4a582",
        "loss_joint_cam":    "#4393c3",
        "loss_joint_proj":   "#92c5de",
    }

    x = list(range(len(epochs)))
    x_labels = [str(e) for e in epochs]

    fig, axs = plt.subplots(2, 2, figsize=(6, 5))

    for row, pair in enumerate(layout):
        for col, key in enumerate(pair):
            ax = axs[row][col]
            if key is None:
                ax.set_visible(False)
                continue
            ax.plot(x, means[key], marker="o", markersize=4,
                    linewidth=1.8, color=colors[key])
            ax.set_title(titles[key], fontsize=10,
                         fontweight="bold" if key == "loss_total" else "normal")
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel("Loss", fontsize=9)
            ax.tick_params(axis="both", labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


def main():
    smooth = 50
    output = "training_loss.png"

    records = parse_log(LOG_PATHS)
    print(f"Parsed {len(records)} log entries across epochs: "
          f"{sorted(set(r['epoch'] for r in records))}")
    plot_losses(records, smooth_window=smooth, output_path=output)
    plot_paper_total(records, output_path="paper_loss_total.pdf")
    plot_paper_components(records, output_path="paper_loss_components.pdf")


if __name__ == "__main__":
    main()
