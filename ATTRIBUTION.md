# Attribution

## Source Code

### Andrej Karpathy — llm.c

Karpathy's [llm.c](https://github.com/karpathy/llm.c) served as the starting point for the training infrastructure. The original files used were `train_gpt2.py` (the main pretraining script, ~861 lines), `dev/data/fineweb.py` (the dataset download and tokenization script), and `dev/data/hellaswag/eval_gpt2.py` (the HellaSwag evaluation script). For this project, the single-file training script was restructured into a modular `src/components/` directory where the model definition (`model.py`), dataloader (`dataloader.py`), attention variants (`attention.py`), and utilities (`utils.py`) are each in their own file. Code that was not needed for this project was removed, including serialization utilities and multi-node distributed training support. The most significant structural change was replacing the original `CausalSelfAttention` module with a dispatched system that selects between five distinct attention score functions at model construction time. The resulting codebase is ~330 lines across the model and training components, compared to ~861 in the original.

### deep-spin/entmax

The [deep-spin/entmax](https://github.com/deep-spin/entmax) library provided the mathematical foundation for the sparsemax and 1.5-entmax algorithms in `src/components/attention.py`. Both `SparsemaxFunction` and `Entmax15Function` were reimplemented from scratch with all math inlined directly rather than using helper functions. The optional partial-sort optimization is not included, the `nn.Module` wrappers are dropped, and the backward passes are written to be compact and tightly integrated into the multi-head causal attention setting. The resulting implementation is ~120 lines compared to the original ~280. The mathematical algorithm is the same but the implementation is original to this project.

## External Libraries

| Library | Version | Use |
|---|---|---|
| PyTorch | 2.10.0 | model, training loop, autograd |
| tiktoken | 0.12.0 | GPT-2 tokenizer |
| numpy | 2.4.3 | data sharding, log parsing |
| matplotlib | 3.10.8 | figure generation |
| seaborn | 0.13.2 | attention map and sparsity heatmaps |
| datasets | 4.8.4 | FineWeb download |
| tqdm | 4.67.3 | progress bars |
| requests | 2.33.1 | data download utility |

## Datasets

**FineWeb-10B** ([HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)) — ~9.8B tokens used for pretraining all five model variants. The tokenization script (`data/fineweb.py`) is adapted from Karpathy's llm.c with minor modifications.

**HellaSwag** ([Zellers et al., 2019](https://arxiv.org/abs/1905.07830)) — The validation split (10,042 examples) was used to evaluate commonsense reasoning accuracy. The `data.jsonl` file is included in the repository under `data/hellaswag/`.

## References

- Karpathy, A. (2024). llm.c. https://github.com/karpathy/llm.c
- Peters, B., Niculae, V., & Martins, A. F. T. (2019). Sparse Sequence-to-Sequence Models. *ACL 2019*. https://github.com/deep-spin/entmax
- Wortsman, M., et al. (2023). Replacing Softmax with ReLU in Vision Transformers. *arXiv:2309.08586*. (Inspiration for the Dynamic ReLU and Dynamic ReLU² attention variants.)
- Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., & Choi, Y. (2019). HellaSwag: Can a Machine Really Finish Your Sentence? *ACL 2019*.

## AI-generated Code

### src/eval_gpt2.py (line 29)

When I ran the evaluation script I got an error I wasn't familiar with where `torch.load` was refusing to deserialize the `GPTConfig` object saved inside the checkpoint. I asked Claude what the error meant and it explained it was a PyTorch 2.4+ security restriction that blocks loading unrecognized Python objects by default. It suggested this fix, which tells PyTorch it is safe to deserialize `GPTConfig` from a checkpoint file:

```python
torch.serialization.add_safe_globals([GPTConfig])
```

Everything else in the function is adapted from Karpathy.

### src/plot_logs.py (lines 87–92)

The raw training loss logs have ~18k data points and are too noisy to read directly. I asked Claude how training curves are typically smoothed in research papers and it suggested a centered moving average using `np.convolve`. The window size (550 steps) and the decision to apply it to both training loss and gradient norm were my own choices.

```python
def smooth(y, window):
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y, kernel, mode="valid")
    offset = window // 2
    return offset, y_smooth
```
