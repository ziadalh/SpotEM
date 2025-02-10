#!/bin/bash

EFFICIENCY=0.50
SAMPLER_LOSS_COEF=300.0
LR=0.001
EXPT_ROOT=${PWD}/efficiency_${EFFICIENCY}_slc_${SAMPLER_LOSS_COEF}_lr_${LR}

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
    --init_lr $LR \
    --use_feature_sampler \
    --feature_sampler_type "liteeval" \
    --feature_sampler_efficiency $EFFICIENCY \
    --sampler_loss_coef $SAMPLER_LOSS_COEF \
    --sampler_loss_type batch \
    --eval_freq 1
