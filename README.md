# A Comparative Study of Attention Score Functions in GPT-2

This project trains five GPT-2 small models from scratch, each with a distinct attention score function, and compares their effect on validation loss, commonsense reasoning accuracy, and per-head attention sparsity. The five mechanisms are Softmax, Sparsemax, 1.5-Entmax, Dynamic ReLU, and Dynamic ReLU², where Dynamic ReLU and Dynamic ReLU² are novel variants introduced in this project.

## What it Does

My main research question for this project was to investigate how different attention score functions impact model quality and training efficiency in GPT-2 pretraining. To do so, I trained and compared five mechanisms including Softmax, Sparsemax, 1.5-Entmax, Dynamic ReLU, and Dynamic ReLU², the last two being novel variants introduced in this project. All five were trained from scratch on the FineWeb-10B dataset (~9.8B tokens over 18,865 steps) using an identical GPT-2 small architecture (124M parameters) and training configuration, making the attention mechanism the only variable. Models are evaluated on validation cross-entropy loss, HellaSwag commonsense reasoning accuracy, per-layer per-head attention sparsity, and training throughput. Results show that 1.5-Entmax achieves the best HellaSwag normalized accuracy (30.23%) while Dynamic ReLU² achieves the lowest validation loss (3.268), and that sparse attention mechanisms like Sparsemax produce dramatically sparser attention patterns (98.3% zeros on average) compared to the dense Softmax baseline.

### Attention Score Functions

All five share the same scaled dot-product pre-step $s_{ij} = q_i \cdot k_j / \sqrt{d_k}$ with causal masking and differ only in how those raw scores are mapped to attention weights.

- **Softmax** $\quad a_i = e^{s_i} / \sum_j e^{s_j}$
- **Sparsemax** (Martins & Astudillo, 2016) $\quad a_i = (s_i - \tau)_+$, $\tau$ has to be such that $\sum_i a_i = 1$
- **1.5-Entmax** (Peters et al., 2019) $\quad a_i = (s_i / 2 - \tau)_+^2$ , $\tau$ has to be such that $\sum_i a_i = 1$
- **Dynamic ReLU**

  $$a_i = \frac{\text{ReLU}(s_i - \max(s) + \sigma(\beta))}{\sum_j \text{ReLU}(s_j - \max(s) + \sigma(\beta))}$$

- **Dynamic ReLU²**

  $$a_i = \frac{\text{ReLU}\left(\dfrac{s_i - \max(s)}{2} + \sigma(\beta)\right)^{2}}{\sum_j \text{ReLU}\left(\dfrac{s_j - \max(s)}{2} + \sigma(\beta)\right)^{2}}$$

## Quick Start

**1. Set up the environment**
```bash
conda env create -f environment.yml
conda activate cs372
```

**2. Download and tokenize the data**
```bash
python data/fineweb.py --type classic --version 10B
```
This downloads FineWeb-10B from HuggingFace and writes tokenized `.bin` shards to `data/fineweb10B/`.

**3. Train a model**

Edit `run_training.sh` to set `ATTN_TYPE` to one of: `softmax`, `sparsemax`, `entmax15`, `dynamic_relu`, `dynamic_relu_square`. Then run:
```bash
bash run_training.sh
```
Logs are written to `src/logs/` and checkpoints to `models/ckpts/`.

**4. Evaluate on HellaSwag**
```bash
python src/eval_gpt2.py --model_path models/ckpts/ckpt_softmax.pt
```

**5. Generate figures**
```bash
# training curves
python src/plot_logs.py

# per-layer per-head sparsity heatmaps
python src/sparsity.py

# attention maps for a sample phrase
python src/attention_map.py
```

**6. Compile the ablation table**

Requires a LaTeX distribution. See [SETUP.md](SETUP.md) for installation instructions.
```bash
cd src/figures && pdflatex ablation_table.tex
```

Figures are saved to `src/figures/`.

## Video Links

- **Demo:** https://youtu.be/_fBzVJCATdE
- **Technical Walkthrough:** https://youtu.be/WV6HIvYXVQI

They are both also in the videos folder! Added Youtube link just in case or if it's easier!

## Results

All five models were trained identically (GPT-2 small, 18,865 steps, FineWeb-10B, cosine LR schedule with warmup, AdamW with weight decay 0.1, gradient clipping 1.0, bfloat16).

| Attention | Val Loss | HellaSwag Acc Norm | Mean Sparsity | Throughput |
|---|---|---|---|---|
| Softmax | 3.2908 | 30.06% | 0.00% | 363k tok/s |
| Sparsemax | 3.2960 | 29.76% | 98.29% | 171k tok/s |
| 1.5-Entmax | 3.2780 | **30.23%** | 93.62% | 161k tok/s |
| Dynamic ReLU | 3.3052 | 29.64% | 91.95% | 259k tok/s |
| Dynamic ReLU² | **3.2680** | 29.89% | 88.39% | 254k tok/s |
| OpenAI GPT-2\* | ≈3.29 | 29.40% | 0.00% | N/A |

\* original OpenAI checkpoint evaluated on FineWeb val and HellaSwag by Karpathy (2024); val loss read approximately from training curve in llm.c discussion #481.

![Training curves across all five attention mechanisms](src/figures/training_curves.png)

All five mechanisms converge to comparable validation loss, spanning a narrow range from 3.268 to 3.305. Dynamic ReLU², a sparse mechanism, achieves the lowest validation loss of any variant including the softmax baseline, suggesting that concentrating attention onto the highest-scoring tokens provides a useful inductive bias during pretraining at this scale. 1.5-Entmax achieves the best HellaSwag normalized accuracy (30.23%), indicating that intermediate sparsity may transfer better to downstream reasoning. Throughput costs vary significantly across mechanisms and are reported in the table above.

![Per-layer per-head attention sparsity across all five models](src/figures/sparsity_heatmaps.svg)

The sparsity heatmaps show that different mechanisms produce structurally different attention patterns across layers and heads. Sparsemax is the most aggressive, with 98.3% zeros on average, attending to fewer than 2% of tokens per query position in the later layers. The Dynamic ReLU variants show high variance across heads, with some heads remaining nearly dense while others approach the sparsity of Sparsemax. This variation comes from the learned per-head β parameter, which adapts the effective attention threshold independently for each head. The central tradeoff is made visible here. Sparse mechanisms yield more interpretable and concentrated attention patterns, but the projection operations that produce those distributions carry a substantial throughput cost relative to softmax.

![Attention maps for a sample phrase, one representative head per model](src/figures/attention_map.svg)

Each map shown above was drawn from a single representative head and layer for each model variant. The heads were chosen independently to illustrate what each mechanism looks like qualitatively and are not selected to represent the same functional role across models. Within any single model, different heads and layers specialize in structurally different patterns, so cross-model comparison at the level of individual heads is limited. The most visible distinction is density. Softmax distributes weight broadly across the context, while the sparse mechanisms produce visibly concentrated patterns with large zero regions, attending to a small fraction of positions even for long sequences.

## Novel Contributions and Comparison to Documented Baseline

Dynamic ReLU and Dynamic ReLU² are attention mechanisms introduced in this project. The starting point was [Wortsman et al. (2023)](https://arxiv.org/abs/2309.08586), who showed that replacing the softmax exponential with a ReLU in vision transformers can maintain performance while producing sparse attention. Their formulation uses a fixed shift of $1/\text{sequence\_length}$. This project adapts and extends that idea to GPT-2 language model pretraining in two ways: the fixed shift is replaced by $\sigma(\beta) = \text{sigmoid}(\beta)$, where $\beta$ is a scalar learned independently by each attention head, allowing different heads to adapt their own sparsity level during training; and Dynamic ReLU² further scales the dot products by 0.5 before the shift and squares the surviving scores before renormalization, concentrating weight more heavily on the highest-scoring tokens. Neither variant appears in Wortsman et al. (2023).

The OpenAI GPT-2 baseline numbers used for comparison (val loss ≈3.29, HellaSwag 29.40%) are from Karpathy (2024), who evaluated the original OpenAI GPT-2 checkpoint on FineWeb val and HellaSwag and reported results in his llm.c codebase (https://github.com/karpathy/llm.c/discussions/481). The val loss of ≈3.29 is read approximately from the training curve plot in that discussion, while the HellaSwag figure of 29.40% is stated explicitly there. Two of the five variants trained in this project outperform that baseline on both metrics. Dynamic ReLU² achieves a val loss of 3.2680 and HellaSwag accuracy of 29.89%, and 1.5-Entmax achieves a val loss of 3.2780 and HellaSwag accuracy of 30.23%, compared to ≈3.29 and 29.40% for the OpenAI checkpoint.

## Individual Contributions

Solo project!