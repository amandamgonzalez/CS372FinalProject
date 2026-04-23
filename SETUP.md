# Setup Instructions

## Requirements

- Linux or macOS
- conda
- Training and evaluation require a CUDA-enabled GPU. Figure generation can be run on CPU.

Approximate training time per variant on an NVIDIA H100 80GB:

| Attention | Time |
|---|---|
| Softmax | ~7.5 hours |
| Dynamic ReLU | ~10.5 hours |
| Dynamic ReLU² | ~10.8 hours |
| Sparsemax | ~16 hours |
| 1.5-Entmax | ~17 hours |

## 1. Clone the Repository

```bash
git clone <repo-url>
cd CS372FinalProject
```

## 2. Create the conda Environment

```bash
conda env create -f environment.yml
conda activate cs372
```

## 3. Download and Tokenize the Data

FineWeb-10B is a public dataset and does not require authentication. Run from the project root:

```bash
python data/fineweb.py --type classic --version 10B
```

Optionally, set a HuggingFace token to speed up the download:

```bash
export HF_TOKEN=your_token_here
python data/fineweb.py --type classic --version 10B
```

This writes tokenized `.bin` shards to `data/fineweb10B/`, producing one validation shard and ~103 training shards of ~100M tokens each. The download requires approximately 50GB of disk space.

## 4. Train a Model

All scripts must be run from the project root. Edit `run_training.sh` and set `ATTN_TYPE` to one of: `softmax`, `sparsemax`, `entmax15`, `dynamic_relu`, `dynamic_relu_square`. Then run:

```bash
bash run_training.sh
```

Logs are written to `src/logs/main_<attn_type>.log` and the final checkpoint to `models/ckpts/ckpt_<attn_type>.pt`. To reproduce all five runs, repeat with each attention type.

### Key Arguments

All hyperparameters are taken directly from Karpathy's llm.c GPT-2 pretraining configuration. The only argument that changes between runs is `ATTN_TYPE`. Do not modify the other values if you want results comparable to those reported in this project.

| Argument | Value | Description |
|---|---|---|
| `ATTN_TYPE` | `softmax` | The only variable between runs. One of: `softmax`, `sparsemax`, `entmax15`, `dynamic_relu`, `dynamic_relu_square` |
| `--total_batch_size` | 524288 | ~0.5M tokens per optimizer step (gradient accumulation steps are calculated automatically) |
| `--num_iterations` | 18865 | Approximately one full pass over FineWeb-10B |
| `--learning_rate` | 0.0006 | Peak learning rate with cosine decay to zero |
| `--warmup_iters` | 700 | Linear warmup steps before cosine decay begins |
| `--weight_decay` | 0.1 | AdamW weight decay |
| `--compile` | 1 | Enables `torch.compile` for faster training |
| `--device` | `cuda` | Change to `cpu` if no GPU is available, but training will be impractically slow |

## 5. Evaluate on HellaSwag

```bash
python src/eval_gpt2.py --model_path models/ckpts/ckpt_softmax.pt
```

Expected output of `acc_norm` should be around 0.300 for the softmax model.

## 6. Generate Figures

```bash
python src/plot_logs.py
python src/sparsity.py
python src/attention_map.py
```

Figures are saved to `src/figures/`.

## 7. Compile the Ablation Table

The following instructions are for macOS. Installation will differ on other operating systems.

```bash
brew install --cask basictex
sudo tlmgr install latexmk
echo 'export PATH="/Library/TeX/texbin:$PATH"' >> ~/.zprofile
```

Open a new terminal, then run from the project root:
```bash
cd src/figures && pdflatex ablation_table.tex
```

The compiled table is saved to `src/figures/ablation_table.pdf`.
