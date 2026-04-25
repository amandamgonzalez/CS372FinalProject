import os
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from components.model import GPT, GPTConfig
from components.dataloader import DistributedDataLoader


labels = {
    "softmax":             "Softmax",
    "sparsemax":           "Sparsemax",
    "dynamic_relu":        "Dynamic ReLU",
    "entmax15":            "1.5-Entmax",
    "dynamic_relu_square": "Dynamic ReLU²",
}


def load_model(ckpt_path, device):
    torch.serialization.add_safe_globals([GPTConfig])
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = GPT(checkpoint["config"])
    state_dict = checkpoint["model"]
    state_dict = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.eval().to(device)


@torch.no_grad()
def measure(ckpt_path, val_bin, B, T, n_steps, device):
    model = load_model(ckpt_path, device)
    n_layer = model.config.n_layer
    n_head = model.config.n_head

    zero_count = torch.zeros(n_layer, n_head, dtype=torch.float64)
    valid_count = torch.zeros(n_layer, n_head, dtype=torch.float64)

    handles = []
    for li, block in enumerate(model.transformer.h):
        def make_hook(layer_idx):
            def hook(*args):
                _, att = args[-1]
                # use only the last query row — the only position with all T keys visible
                last = att[:, :, -1, :].cpu()  # (B, H, T)
                B_, H, T_ = last.shape
                zero_count[layer_idx] += (last == 0).sum(dim=(0, 2)).double()
                valid_count[layer_idx] += float(B_ * T_)
            return hook
        handles.append(block.attn.register_forward_hook(make_hook(li)))

    val_loader = DistributedDataLoader(val_bin, B, T, process_rank=0, num_processes=1)
    for _ in tqdm(range(n_steps), desc=ckpt_path):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        model(x, y, return_logits=False)

    for h in handles:
        h.remove()

    return (zero_count / valid_count.clamp(min=1)).numpy()


def plot_heatmaps(matrices, out_path):
    sns.set_theme(context="paper", style="white", font_scale=1.0)
    keys = list(matrices.keys())
    n = len(keys)
    fig, axes = plt.subplots(
        1, n, figsize=(3.4 * n + 1.0, 4.2), sharex=True, sharey=True,
        constrained_layout=True,
    )
    if n == 1:
        axes = np.array([axes])

    vmin, vmax = 0.0, 1.0
    for ax, key in zip(axes.flat, keys):
        mat = matrices[key]
        sns.heatmap(
            mat, ax=ax, vmin=vmin, vmax=vmax, cmap="magma",
            cbar=False, square=True, xticklabels=2, yticklabels=2,
        )
        ax.set_title(
            f"{labels.get(key, key)} / Mean: {mat.mean():.2f}",
            pad=3, size=10,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes.flat:
        ax.set_xlabel("Heads", size=13)
    axes.flat[0].set_ylabel("Layers", size=13)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="magma")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.04)
    cbar.set_label("Sparsity", size=13, labelpad=12)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(out_path, bbox_inches="tight")
    svg_path = os.path.splitext(out_path)[0] + ".svg"
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="models/ckpts")
    p.add_argument("--val_bin", default="data/fineweb10B/fineweb_val_*.bin")
    p.add_argument("--out_json", default="src/logs/sparsity.json")
    p.add_argument("--out_pdf", default="src/figures/sparsity_heatmaps.pdf")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--n_steps", type=int, default=64)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.set_float32_matmul_precision("highest")

    runs = {
        "softmax":             os.path.join(args.ckpt_dir, "ckpt_softmax.pt"),
        "sparsemax":           os.path.join(args.ckpt_dir, "ckpt_sparsemax.pt"),
        "dynamic_relu":        os.path.join(args.ckpt_dir, "ckpt_dynamic_relu.pt"),
        "entmax15":            os.path.join(args.ckpt_dir, "ckpt_entmax15.pt"),
        "dynamic_relu_square": os.path.join(args.ckpt_dir, "ckpt_dynamic_relu_square.pt"),
    }

    matrices = {}
    for attn_type, ckpt_path in runs.items():
        if not os.path.exists(ckpt_path):
            print(f"skipping {attn_type}, no checkpoint found at {ckpt_path}")
            continue
        mat = measure(ckpt_path, args.val_bin, args.batch_size, args.seq_len, args.n_steps, args.device)
        matrices[attn_type] = mat
        print(f"\n{attn_type} overall mean sparsity {mat.mean():.4f}")
        for li, row in enumerate(mat):
            print(f"  layer {li:2d} mean sparsity {row.mean():.4f}")
    matrices["softmax"] *= 0

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(
            {k: {"per_layer_per_head": v.tolist(), "overall_mean": float(v.mean())}
             for k, v in matrices.items()},
            f, indent=2,
        )
    print(f"\nsaved sparsity results to {args.out_json}")

    if matrices:
        plot_heatmaps(matrices, args.out_pdf)
        print(f"saved heatmap to {args.out_pdf}")
        print(f"saved heatmap to {os.path.splitext(args.out_pdf)[0]}.svg")


if __name__ == "__main__":
    main()
