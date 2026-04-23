import os
import argparse

import numpy as np
import torch
import tiktoken

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from components.model import GPT, GPTConfig


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
def get_attention_map(ckpt_path, token_ids, layer, head, device):
    model = load_model(ckpt_path, device)

    captured = {}
    def hook(*args):
        _, att = args[-1]
        captured["att"] = att.detach().cpu()

    handle = model.transformer.h[layer].attn.register_forward_hook(hook)
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    model(x, return_logits=False)
    handle.remove()

    return captured["att"][0, head].numpy()


def plot_attention_maps(matrices, tokens, out_path):
    sns.set_theme(context="paper", style="white", font_scale=1.0)
    keys = list(matrices.keys())
    n = len(keys)
    fig, axes = plt.subplots(
        1, n, figsize=(4.2 * n + 1.0, 4.6),
        sharex=True, sharey=True, constrained_layout=True,
    )
    if n == 1:
        axes = np.array([axes])

    vmin, vmax = 0.0, 1.0
    for ax, key in zip(axes.flat, keys):
        mat = matrices[key]
        sns.heatmap(
            mat, ax=ax, vmin=vmin, vmax=vmax, cmap="magma",
            cbar=False, square=True,
            xticklabels=tokens, yticklabels=tokens,
        )
        ax.set_title(labels.get(key, key), pad=4, size=11)
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.set_xlabel("")
        ax.set_ylabel("")

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="magma")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.04)
    cbar.set_label("Attention probability", size=12, labelpad=10)
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(out_path, bbox_inches="tight")
    svg_path = os.path.splitext(out_path)[0] + ".svg"
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def parse_pair(s):
    layer, head = s.split(",")
    return int(layer), int(head)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="models/ckpts")
    p.add_argument("--out_pdf", default="src/figures/attention_map.pdf")
    p.add_argument("--phrase", default="She told her sister that she was wrong.")
    p.add_argument("--softmax",             type=parse_pair, default=(3, 3))
    p.add_argument("--sparsemax",           type=parse_pair, default=(4, 6))
    p.add_argument("--dynamic_relu",        type=parse_pair, default=(3, 10))
    p.add_argument("--entmax15",            type=parse_pair, default=(3, 3))
    p.add_argument("--dynamic_relu_square", type=parse_pair, default=(3, 1))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.set_float32_matmul_precision("highest")

    enc = tiktoken.get_encoding("gpt2")
    token_ids = enc.encode(args.phrase)
    tokens = [enc.decode([tid]) for tid in token_ids]
    print(f"phrase: {args.phrase!r}")
    print(f"tokens ({len(tokens)}): {tokens}")

    runs = {
        "softmax":             (os.path.join(args.ckpt_dir, "ckpt_softmax.pt"),             args.softmax),
        "sparsemax":           (os.path.join(args.ckpt_dir, "ckpt_sparsemax.pt"),           args.sparsemax),
        "dynamic_relu":        (os.path.join(args.ckpt_dir, "ckpt_dynamic_relu.pt"),        args.dynamic_relu),
        "entmax15":            (os.path.join(args.ckpt_dir, "ckpt_entmax15.pt"),            args.entmax15),
        "dynamic_relu_square": (os.path.join(args.ckpt_dir, "ckpt_dynamic_relu_square.pt"), args.dynamic_relu_square),
    }

    matrices = {}
    for attn_type, (ckpt_path, (layer, head)) in runs.items():
        if not os.path.exists(ckpt_path):
            print(f"skipping {attn_type}, no checkpoint found at {ckpt_path}")
            continue
        mat = get_attention_map(ckpt_path, token_ids, layer, head, args.device)
        matrices[attn_type] = mat
        print(f"{attn_type} layer {layer} head {head} row sums {mat.sum(axis=-1)}")

    if not matrices:
        print("no checkpoints found, nothing to plot")
        return

    os.makedirs(os.path.dirname(args.out_pdf), exist_ok=True)
    plot_attention_maps(matrices, tokens, args.out_pdf)
    print(f"saved attention map to {args.out_pdf}")
    print(f"saved attention map to {os.path.splitext(args.out_pdf)[0]}.svg")


if __name__ == "__main__":
    main()
