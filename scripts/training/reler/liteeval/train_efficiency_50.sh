#!/bin/bash

export CUDA_VISIBLE_DEVICES="$1"
EFFICIENCY=0.5
LOSS_COEF=300.0
LR=0.000004
LR_SCALE=100.0

LOSS_TYPE="batch"
EPOCHS=200

EXPT_ROOT=$PWD
cd $SPOTEM_ROOT/RELER

ctx_mode=video_tef
v_feat_types=egovlp_clip_imagenet
t_feat_type=clip
results_root=$EXPT_ROOT/efficiency_${EFFICIENCY}_slc_${LOSS_COEF}_lr_${LR}_${LR_SCALE}
task_type=nlq_official_v1

######## setup video+text features
feat_root=features

# video features
v_feat_dim=0
vsldataset_fv="egovlp"
v_feat_dirs=()
if [[ ${v_feat_types} == *"egovlp"*  ]]; then
  v_feat_dirs+=(../data/features/${task_type}/egovlp)
    (( v_feat_dim += 256  ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
  vsldataset_fv="egovlp"
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(../data/features/${task_type}/CLIP_video)
  (( v_feat_dim += 512 ))
fi
if [[ ${v_feat_types} == *"object"* ]]; then
  v_feat_dirs+=(../data/features/${task_type}/object_features)
  (( v_feat_dim += 256 ))
fi
if [[ ${v_feat_types} == *"room"* ]]; then
  v_feat_dirs+=(../data/features/${task_type}/room_features)
  (( v_feat_dim += 256 ))
fi
if [[ ${v_feat_types} == *"interaction"* ]]; then
  v_feat_dirs+=(../data/features/${task_type}/interaction_features)
  (( v_feat_dim += 256 ))
fi
if [[ ${v_feat_types} == *"imagenet"* ]]; then
  v_feat_dirs+=(../data/features/${task_type}/imagenet_features)
  (( v_feat_dim += 1280 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=../data/features/${task_type}/CLIP_text
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

# multi_scale param
scale_list=()
scale_list+=(2)
scale_list+=(3)
scale_list+=(4)
scale_list+=(5)
scale_list+=(6)

# hyperparameters
MAX_V_LEN=600
HIDDEN_SIZE=256
N_CROSS_ENCODER_LAYERS=3
DROPOUT=0.1
USE_SW=0
USE_VS=0
VS_PROB=0.5
CONTRASTIVE_HIDDEN_SIZE=64
EXP_NAME=vlen600_slowfast

#### training
bsz=128

sw_len_ratio=()
sw_len_ratio+=(0.4)
sw_len_ratio+=(0.8)

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(($RANDOM+1000000000))
    echo $(($num%$max+$min))

}
export MAIN_PORT=$(rand 1024 2048)
export MASTER_PORT=$(rand 1024 2048)

PYTHONPATH=$PYTHONPATH:. python -W ignore -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=4 \
    --master_port $MASTER_PORT \
     ms_cm/train_ego4d_slowfast.py \
        --ctx_mode ${ctx_mode} \
        --v_feat_dirs ${v_feat_dirs[@]} \
        --numscale_list ${scale_list[@]} \
        --v_feat_dim ${v_feat_dim} \
        --t_feat_dir ${t_feat_dir} \
        --t_feat_dim ${t_feat_dim} \
        --bsz ${bsz} \
        --results_root ${results_root} \
        --vsldataset_task ${task_type} \
        --vsldataset_fv ${vsldataset_fv} \
        --no_pin_memory \
        --lr $LR \
        --num_workers 10 \
        --vsldataset_num_workers 4 \
        --vslnet_datapath ../data/ \
        --vslnet_dataset_save_dir ./vslnet_dataset_savepath \
        --eval_gt_json ../data/nlq_val.json \
        --no_aux_loss \
        --cross_first \
        --max_v_l $MAX_V_LEN \
        --lw_saliency 1 \
        --lw_highlight 20 \
        --enc_layers 0 \
        --hidden_dim $HIDDEN_SIZE \
        --v_hidden_size $HIDDEN_SIZE \
        --bi_hidden_size $HIDDEN_SIZE \
        --hidden_size $HIDDEN_SIZE \
        --num_cross_encoder_layers $N_CROSS_ENCODER_LAYERS \
        --dropout $DROPOUT \
        --v_hidden_dropout_prob $DROPOUT \
        --hidden_dropout_prob $DROPOUT \
        --v_attention_probs_dropout_prob $DROPOUT \
        --attention_probs_dropout_prob $DROPOUT \
        --vslnet_thres_in_train 0 \
        --use_sw $USE_SW \
        --sw_len_ratio ${sw_len_ratio[@]} \
        --use_vs $USE_VS \
        --vs_prob $VS_PROB \
        --video_frame_contrastive_loss \
        --video_frame_contrastive_loss_coef 1 \
        --contrastive_hdim $CONTRASTIVE_HIDDEN_SIZE \
        --exp_id $EXP_NAME \
        --use_feature_sampler \
        --feature_sampler_type liteeval \
        --feature_sampler_efficiency $EFFICIENCY \
        --feature_mask_idxs 0 767 \
        --sampler_loss_coef $LOSS_COEF \
        --sampler_loss_type $LOSS_TYPE \
        --resume ../pretrained_models/reler/uniform_50.t7 \
        --lr $LR \
        --lr_scale_sampler $LR_SCALE \
        --n_epoch $EPOCHS \
        --eval_epoch_interval 3
