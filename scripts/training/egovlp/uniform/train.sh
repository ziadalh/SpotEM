#!/bin/bash

EFFICIENCY=$2
EXPT_ROOT=${PWD}/efficiency_${EFFICIENCY}

cd $SPOTEM_ROOT


CUDA_VISIBLE_DEVICES="$1" python VSLNet/main.py \
    --task nlq_official_v1 \
    --predictor egovlp-distilbert \
    --dim 128 \
    --mode train \
    --video_feature_dim 1536 \
    --feature_mask_idxs 0 255 \
    --max_pos_len 128 \
    --epochs 200 \
    --fv 'egovlp+imagenet' \
    --num_workers 64 \
    --model_dir $EXPT_ROOT/checkpoints/ \
    --eval_gt_json "data/nlq_val.json" \
    --tb_log_dir $EXPT_ROOT/tb \
    --log_to_tensorboard "baseline" \
    --remove_empty_queries_from train val \
    --batch_size 128 \
    --init_lr 0.001 \
    --use_feature_sampler \
    --feature_sampler_type "uniform" \
    --feature_sampler_efficiency $EFFICIENCY \
    --eval_freq 1
