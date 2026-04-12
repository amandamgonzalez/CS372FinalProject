import os
import json
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from components.model import GPT, GPTConfig


enc = tiktoken.get_encoding("gpt2")


def iterate_examples():
    # there are 10,042 examples in total in val
    with open(os.path.join("dev/data/hellaswag/data.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

@torch.no_grad()
def evaluate(model_path):

    device = "cuda"
    torch.set_float32_matmul_precision('highest')

    torch.serialization.add_safe_globals([GPTConfig])
    checkpoint = torch.load(model_path, map_location="cpu")
    model = GPT(checkpoint["config"])
    state_dict = checkpoint["model"]
    # torch.compile wraps the model and prefixes keys with "_orig_mod."
    unwanted_prefix = "_orig_mod."
    state_dict = {
        (k[len(unwanted_prefix):] if k.startswith(unwanted_prefix) else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.eval().to(device)

    num_correct_norm = 0
    num_correct = 0
    num_total = 0

    pbar = tqdm(iterate_examples(), total=10042)
    for example in pbar:
        ctx_tokens = enc.encode(example["ctx"])
        label = example["label"]

        candidate_losses = []
        for ending in example["endings"]:
            end_tokens = enc.encode(" " + ending)  # prepend " " per GPT-2 tokenizer convention
            tokens = torch.tensor(
                [ctx_tokens + end_tokens], dtype=torch.long, device=device
            )  # (1, T)

            # pass targets to force full-sequence logits (without targets the model only
            # computes logits for the last position as an inference-time optimization)
            logits, _ = model(tokens, targets=tokens)  # (1, T, vocab_size)

            shift_logits = logits[0, :-1, :]   # (T-1, vocab_size)
            shift_tokens = tokens[0, 1:]        # (T-1,)
            token_losses = F.cross_entropy(shift_logits, shift_tokens, reduction='none')

            # keep only the losses for ending tokens
            # in the shifted sequence, ending tokens start at position n-1
            # (position n-1 predicts the first ending token at position n)
            n = len(ctx_tokens)
            candidate_losses.append(token_losses[n - 1:])

        sum_losses  = [l.sum().item()  for l in candidate_losses]
        mean_losses = [l.mean().item() for l in candidate_losses]
        pred      = min(range(4), key=lambda i: sum_losses[i])
        pred_norm = min(range(4), key=lambda i: mean_losses[i])  # length-normalized (standard)

        num_total += 1
        num_correct      += int(pred      == label)
        num_correct_norm += int(pred_norm == label)
        pbar.set_postfix(acc=f"{num_correct/num_total:.4f}", acc_norm=f"{num_correct_norm/num_total:.4f}")

        # if num_total < 10:
        #     print("---")
        #     print(f"Context:\n {example['ctx']}")
        #     print("Endings:")
        #     for i, (end, ml) in enumerate(zip(example["endings"], mean_losses)):
        #         print(f"  {i} (loss: {ml:.4f}) {end}")
        #     print(f"predicted: {pred_norm}, actual: {label}")

    print(f"acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--model_path", type=str, required=True, help="path to model checkpoint")
    args = parser.parse_args()
    evaluate(args.model_path)
