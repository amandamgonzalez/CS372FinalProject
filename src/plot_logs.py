"""
visualization of training logs for gpt-2 attention comparison.
produces a 3-panel figure with validation loss, smoothed training loss, smoothed gradient norm.
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.labelsize":     10,
    "axes.titlesize":     11,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#d0d0d0",
    "grid.linewidth":     0.6,
    "grid.alpha":         1.0,
    "lines.linewidth":    1.6,
    "figure.dpi":         150,
})

colors = {
    "softmax":             "#3B0F70",
    "sparsemax":           "#8C2981",
    "dynamic_relu":        "#DE4968",
    "entmax15":            "#FB8861",
    "dynamic_relu_square": "#FEC387",
}
labels = {
    "softmax":             "Softmax",
    "sparsemax":           "Sparsemax",
    "dynamic_relu":        "Dynamic ReLU",
    "entmax15":            "1.5-Entmax",
    "dynamic_relu_square": "Dynamic ReLU²",
}
log_files = {
    "softmax":             "src/logs/main_softmax.log",
    "sparsemax":           "src/logs/main_sparsemax.log",
    "dynamic_relu":        "src/logs/main_dynamic_relu.log",
    "entmax15":            "src/logs/main_entmax15.log",
    "dynamic_relu_square": "src/logs/main_dynamic_relu_square.log",
}

smooth_window = 550

def parse_log(path):
    val_steps, val_losses = [], []
    train_steps, train_losses, norms, throughputs = [], [], [], []
    val_re = re.compile(r"^s:(\d+) tel:([\d.]+)")
    train_re = re.compile(
        r"^step\s+(\d+)/\d+\s+\|\s+train loss ([\d.]+)\s+\|\s+norm ([\d.]+)"
        r".*\|\s+\([\d.]+ ms \| (\d+) tok/s\)"
    )

    with open(path) as f:
        for line in f:
            m = val_re.match(line)
            if m:
                val_steps.append(int(m.group(1)))
                val_losses.append(float(m.group(2)))
                continue
            m = train_re.match(line)
            if m:
                step = int(m.group(1))
                train_steps.append(step)
                train_losses.append(float(m.group(2)))
                norms.append(float(m.group(3)))
                if step > 1:  # skip step 1 — always slow due to gpu warmup
                    throughputs.append(float(m.group(4)))

    return (
        np.array(val_steps),   np.array(val_losses),
        np.array(train_steps), np.array(train_losses),
        np.array(norms),       np.mean(throughputs) if throughputs else 0.0,
    )


def smooth(y, window):
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y, kernel, mode="valid")
    # center the smoothed curve on its x range
    offset = window // 2
    return offset, y_smooth


data = {
    name: parse_log(path)
    for name, path in log_files.items()
    if os.path.getsize(path) > 0
}


fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
fig.subplots_adjust(wspace=0.35)

for name, (val_steps, val_losses,
           tr_steps, tr_losses, norms, _) in data.items():

    color = colors[name]
    label = labels[name]

    # validation loss
    axes[0].plot(val_steps, val_losses, color=color, label=label)

    # smoothed training loss
    # applied a centered moving average over `smooth_window` steps using
    # np.convolve with a uniform kernel, then aligning the x-axis to the midpoint of each window
    # this removes per-step noise (~18k raw points) while preserving the curve shape
    offset, tr_smooth = smooth(tr_losses, smooth_window)
    x_smooth = tr_steps[offset : offset + len(tr_smooth)]
    axes[1].plot(x_smooth, tr_smooth, color=color, label=label)

    # smoothed gradient norm
    # same moving average applied for the same reason
    offset, norm_smooth = smooth(norms, smooth_window)
    x_smooth = tr_steps[offset : offset + len(norm_smooth)]
    axes[2].plot(x_smooth, norm_smooth, color=color, label=label)


step_fmt = mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))

panel_titles  = ["Validation Loss", "Training Loss", "Gradient Norm"]
panel_ylabels = ["Cross-Entropy Loss", "Cross-Entropy Loss", "Gradient $\\ell_2$ Norm"]

for ax, title, ylabel in zip(axes, panel_titles, panel_ylabels):
    ax.set_title(title, pad=6)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(step_fmt)
    ax.legend(frameon=False)

# clip loss panels
axes[0].set_ylim(bottom=3.2, top=4)
axes[1].set_ylim(bottom=3.2, top=4)


out_dir = "src/figures"
os.makedirs(out_dir, exist_ok=True)

fig.savefig(os.path.join(out_dir, "training_curves.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "training_curves.png"), bbox_inches="tight", dpi=200)
print(f"saved to {out_dir}/training_curves.{{pdf,png}}")

plt.show()
