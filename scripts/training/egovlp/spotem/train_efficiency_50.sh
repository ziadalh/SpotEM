#!/bin/bash

EFFICIENCY=0.50
SAMPLER_LOSS_COEF=300.0
DISTILL_L_COEF=1.0
DISTILL_F_COEF=1.0
DISTILL_H_COEF=30.0
EXPT_ROOT=$PWD/efficiency_${EFFICIENCY}_slc_${SAMPLER_LOSS_COEF}_dcoefs_${DISTILL_L_COEF}_${DISTILL_F_COEF}_${DISTILL_H_COEF}

cd $SPOTEM_ROOT


CUDA_VISIBLE_DEVICES="$1" python VSLNet/main.py \
    --task nlq_official_v1 \
    --predictor egovlp-distilbert \
    --dim 128 \
    --mode train \
    --video_feature_dim 1024 \
    --feature_mask_idxs 0 255 \
    --max_pos_len 128 \
    --epochs 200 \
    --fv "egovlp+rio" \
    --num_workers 64 \
    --model_dir $EXPT_ROOT/checkpoints/ \
    --eval_gt_json "data/nlq_val.json" \
    --tb_log_dir $EXPT_ROOT/tb \
    --log_to_tensorboard "baseline" \
    --remove_empty_queries_from train val \
    --batch_size 128 \
    --init_lr 0.001 \
    --use_feature_sampler \
    --feature_sampler_type "transformer-v1" \
    --feature_sampler_efficiency ${EFFICIENCY} \
    --sampler_loss_coef ${SAMPLER_LOSS_COEF} \
    --sampler_niters 4 \
    --sampler_loss_type batch \
    --use_full_distillation \
    --distillation_ckpt_path pretrained_models/egovlp/spotem_distill_teacher.t7 \
    --distill_l_coef ${DISTILL_L_COEF} \
    --distill_f_coef ${DISTILL_F_COEF} \
    --distill_h_coef ${DISTILL_H_COEF} \
    --eval_freq 1
