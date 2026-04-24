# A Comparative Study of Attention Score Functions in GPT-2

Designing better attention mechanisms is an active area of research in transformers, with work ranging from efficient approximations to structured sparse alternatives. Most of this work, however, focuses on efficiency or is evaluated on fine-tuning tasks rather than pretraining from scratch. This project trains five GPT-2 small models from scratch, each with a distinct attention score function (Softmax, Sparsemax, 1.5-Entmax, Dynamic ReLU, and Dynamic ReLU²), and systematically evaluates the effect on training dynamics, commonsense reasoning accuracy, and per-head attention sparsity.

## What it Does

This project investigates how replacing the standard softmax attention score function in GPT-2 affects language model training dynamics, downstream task performance, and attention sparsity. Five attention mechanisms are compared: Softmax (baseline), Sparsemax, 1.5-Entmax, Dynamic ReLU, and Dynamic ReLU² (the last two being novel variants introduced in this project, inspired by Wortsman et al. (2023), that replace softmax with a ReLU-based normalization and add a learned per-head scalar parameter (β) which controls the effective attention threshold adaptively across heads). The ReLU² variant further concentrates attention weights on the highest-scoring tokens by squaring post-ReLU scores. 

All five variants are trained from scratch on the FineWeb-10B dataset (~9.8B tokens over 18,865 steps) using an identical GPT-2 small architecture (124M parameters) and training configuration, making the attention mechanism the only variable. Models are evaluated on validation cross-entropy loss, HellaSwag commonsense reasoning accuracy, per-layer per-head attention sparsity, and training throughput. Results show that 1.5-Entmax achieves the best HellaSwag normalized accuracy (30.23%) while Dynamic ReLU² achieves the lowest validation loss (3.268), and that sparse attention mechanisms like Sparsemax produce dramatically sparser attention patterns (98.3% zeros on average) compared to the dense Softmax baseline.

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

- **Demo:** 
- **Technical Walkthrough:** 

## Evaluation

All five models were trained identically (GPT-2 small, 18,865 steps, FineWeb-10B, cosine LR schedule with warmup, AdamW with weight decay 0.1, gradient clipping 1.0, bfloat16).

| Attention | Val Loss | HellaSwag Acc Norm | Mean Sparsity |
|---|---|---|---|
| Softmax | 3.2908 | 30.06% | 0.00% |
| Sparsemax | 3.2960 | 29.76% | 98.29% |
| 1.5-Entmax | 3.2780 | **30.23%** | 93.62% |
| Dynamic ReLU | 3.3052 | 29.64% | 91.95% |
| Dynamic ReLU² | **3.2680** | 29.89% | 88.39% |

**Key findings:**
- Dynamic ReLU² achieves the lowest validation loss (3.2680) despite being a sparse mechanism, suggesting that squaring post-ReLU scores provides a beneficial inductive bias during pretraining.
- 1.5-Entmax achieves the highest HellaSwag normalized accuracy (30.23%), indicating that its intermediate sparsity may generalize better on downstream reasoning tasks.
- All five mechanisms reach comparable validation loss, suggesting sparse attention can match or exceed softmax at this scale without significant degradation.
- Sparsemax is the sparsest mechanism (98.3% zeros), attending to fewer than 2% of tokens on average by the final layers.
- Dynamic ReLU variants produce variable sparsity across heads (some heads nearly dense, others nearly as sparse as sparsemax), suggesting the learned β parameter allows different heads to specialize.
- Training throughput is significantly lower for sparse mechanisms (~172k tok/s for sparsemax vs ~364k tok/s for softmax).

Qualitative attention maps and per-layer sparsity heatmaps are in `src/figures/`.

## Individual Contributions

Solo project!