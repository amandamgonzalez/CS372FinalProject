#!/bin/bash

# while [[ $# -gt 0 ]]; do
#   case $1 in
#     --attn_type) ATTN_TYPE="$2"; shift 2 ;;
#     *) shift ;;  # silently skip unknown args
#   esac
# done

# ATTN_TYPE=${ATTN_TYPE:-softmax}  # default to softmax
ATTN_TYPE=softmax

echo "Using attention type: $ATTN_TYPE"

# python train_gpt2.py ... (all the other arguments the same)
# torchrun --standalone --nproc_per_node=8 train_gpt2.py \
python train_gpt2.py \
    --input_bin "dev/data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "dev/data/fineweb10B/fineweb_val_*.bin" \
    --val_loss_every 250 \
    --sample_every 0 \
    --output_dir logs \
    --ckpts_dir ckpts \
    --model d12 \
    --attn_type $ATTN_TYPE \
    --batch_size 128 \
    --sequence_length 1024 \
    --total_batch_size 524288 \
    --dtype bfloat16 \
    --compile 1 \
    --tensorcores 1 \
    --num_iterations 18865 \
    --weight_decay 0.1 \
    --zero_stage 0 \
    --learning_rate 0.0006 \
    --warmup_iters 700 \
    --learning_rate_decay_frac 0.0 \
    --overfit_single_batch 0