#! /usr/bin/env python
"""
Reading command line options.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse


def read_command_line():
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument(
        "--save_dir",
        type=str,
        default="datasets",
        help="path to save processed dataset",
    )
    parser.add_argument("--task", type=str, default="charades", help="target task")
    parser.add_argument(
        "--eval_gt_json",
        type=str,
        default=None,
        help="Provide GT JSON to evaluate while training",
    )
    parser.add_argument(
        "--fv", type=str, default="new", help="[new | org] for visual features"
    )
    parser.add_argument(
        "--max_pos_len",
        type=int,
        default=128,
        help="maximal position sequence length allowed",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of CPU workers to process the data",
    )
    parser.add_argument(
        "--data_loader_workers",
        type=int,
        default=0,
        help="Number of CPU workers for the dataloader",
    )
    # model parameters
    parser.add_argument("--word_size", type=int, default=None, help="number of words")
    parser.add_argument(
        "--char_size", type=int, default=None, help="number of characters"
    )
    parser.add_argument(
        "--word_dim", type=int, default=300, help="word embedding dimension"
    )
    parser.add_argument(
        "--video_feature_dim",
        type=int,
        default=1024,
        help="video feature input dimension",
    )
    parser.add_argument(
        "--query_feature_dim",
        type=int,
        default=768,
        help="text feature input dimension",
    )
    parser.add_argument(
        "--char_dim",
        type=int,
        default=50,
        help="character dimension, set to 100 for activitynet",
    )
    parser.add_argument("--dim", type=int, default=128, help="hidden size")
    parser.add_argument(
        "--highlight_lambda",
        type=float,
        default=5.0,
        help="lambda for highlight region",
    )
    parser.add_argument("--num_heads", type=int, default=8, help="number of heads")
    parser.add_argument("--drop_rate", type=float, default=0.2, help="dropout rate")
    parser.add_argument(
        "--predictor", type=str, default="rnn", help="[rnn | transformer]"
    )
    parser.add_argument(
        "--egovlp-predictor-ckpt",
        type=str,
        default="/vision/srama/Research/EfficientEpisodicMemory/dependencies/EgoVLP/pretrained_models/egovlp_text_weights.pth",
    )
    # training/evaluation parameters
    parser.add_argument("--gpu_idx", type=str, default="0", help="GPU index")
    parser.add_argument("--seed", type=int, default=12345, help="random seed")
    parser.add_argument("--mode", type=str, default="train", help="[train | test]")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--num_train_steps", type=int, default=None, help="number of training steps"
    )
    parser.add_argument(
        "--init_lr", type=float, default=0.0001, help="initial learning rate"
    )
    parser.add_argument(
        "--clip_norm", type=float, default=1.0, help="gradient clip norm"
    )
    parser.add_argument(
        "--warmup_proportion", type=float, default=0.0, help="warmup proportion"
    )
    parser.add_argument(
        "--extend", type=float, default=0.1, help="highlight region extension"
    )
    parser.add_argument(
        "--period", type=int, default=100, help="training loss print period"
    )
    parser.add_argument(
        "--text_agnostic",
        dest="text_agnostic",
        action="store_true",
        default=False,
        help="Text agnostic model; random text features",
    )
    parser.add_argument(
        "--video_agnostic",
        dest="video_agnostic",
        action="store_true",
        default=False,
        help="Video agnostic model; random video features",
    )
    parser.add_argument(
        "--resume", type=str, default="", help="resume from a pretrained checkpoint"
    )  # noqa
    parser.add_argument(
        "--model_dir",
        type=str,
        default="checkpoints",
        help="path to save trained model weights",
    )
    parser.add_argument("--model_name", type=str, default="vslnet", help="model name")  # noqa
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="set to the last `_xxx` in ckpt repo to eval results",
    )
    parser.add_argument(
        "--log_to_tensorboard",
        type=str,
        default=None,
        help="Comment for the run. Supports multiple runs",
    )
    parser.add_argument(
        "--tb_log_dir",
        type=str,
        default="./runs",
        help="Where the tensorboard logdir is located",
    )
    parser.add_argument(
        "--tb_log_freq",
        type=int,
        default=1,
        help="Log every `tb_log_freq` iterations",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=-1,
        help="Evaluation frequency in epochs (-1 logs twice every epoch)",
    )
    parser.add_argument(
        "--slurm",
        dest="slurm",
        action="store_true",
        default=False,
        help="Schedule with slurm?",
    )
    parser.add_argument(
        "--slurm_wait",
        dest="slurm_wait",
        action="store_true",
        default=False,
        help="Wait for slurm to finish?",
    )
    parser.add_argument(
        "--slurm_partition",
        type=str,
        default="pixar",
        help="Which slurm partition?",
    )
    parser.add_argument(
        "--slurm_constraint",
        type=str,
        default="volta",
        help="Constraint on slurm",
    )
    parser.add_argument(
        "--slurm_gpus",
        type=int,
        default=1,
        help="number of gpus to schedule with slurm",
    )
    parser.add_argument(
        "--slurm_cpus",
        type=int,
        default=10,
        help="number of cpus to schedule with slurm",
    )
    parser.add_argument(
        "--slurm_timeout_min",
        type=int,
        default=720,
        help="How many minutes to schedule",
    )
    parser.add_argument(
        "--slurm_log_folder",
        type=str,
        default="slurm_log",
        help="where to keep slurm logs",
    )
    parser.add_argument(
        "--remove_empty_queries_from",
        type=str,
        nargs="+",
        default=None,
        help="A list of splits to remove empty queries from. Valid values for the list are: ['train', 'val']",  # noqa
    )
    parser.add_argument(
        "--use_egovlp_sent_feats",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--egovlp_sent_feats",
        type=str,
        default="",
        help="path to egovlp sentence features",
    )
    parser.add_argument(
        "--use_feature_sampler",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--feature_sampler_type",
        default="random",
        type=str,
        choices=[
            "random",
            "uniform",
            "zero",
            "all",
            "transformer-v1",
            "liteeval",
            "ocsampler",
        ],
    )
    parser.add_argument(
        "--feature_sampler_efficiency",
        default=0.50,
        type=float,
    )
    parser.add_argument(
        "--sampler_efficiency_levels",
        nargs="+",
        default=[0.90, 0.75, 0.5],
        type=float,
    )
    parser.add_argument(
        "--feature_mask_idxs",
        default=[0, 255],
        type=int,
        nargs=2,
    )
    parser.add_argument(
        "--sampler_loss_coef",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--sampler_loss_type", default="sample", type=str, choices=["sample", "batch"]
    )
    parser.add_argument(
        "--use_split_visual_projection",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--feature_split_points",
        default=[],
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--feature_split_dims",
        default=[],
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--sampler_niters",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--sampler_niters_per_efficiency",
        nargs="+",
        default=[2, 3, 3],
        type=int,
    )
    parser.add_argument(
        "--sampler_mask_prev",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--video_mask_for_sampler_loss",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--disable_stepwise_loss",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_sampler_iteration_indicator",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--sampler_iteration_indicator_dim",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--sampler_initial_tau",
        default=5.0,
        type=float,
    )
    parser.add_argument(
        "--sampler_final_tau",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--sampler_coarse_hdim",
        default=512,
        type=int,
    )
    parser.add_argument(
        "--sampler_fine_hdim",
        default=2048,
        type=int,
    )
    parser.add_argument(
        "--lr_scale_sampler",
        type=float,
        default=1.0,
        help="scaling factor for sampler LR",
    )  # noqa
    parser.add_argument(
        "--eval_efficiency_values",
        type=float,
        nargs="+",
        default=[0.50, 0.75, 0.90],
    )  # noqa
    parser.add_argument(
        "--use_full_distillation",
        action="store_true",
        default=False,
    )  # noqa
    parser.add_argument(
        "--distillation_ckpt_path",
        type=str,
        default="",
    )  # noqa
    parser.add_argument(
        "--distill_l_coef",
        type=float,
        default=1.0,
    )  # noqa
    parser.add_argument(
        "--distill_f_coef",
        type=float,
        default=1.0,
    )  # noqa
    parser.add_argument(
        "--distill_h_coef",
        type=float,
        default=1.0,
    )  # noqa
    parser.add_argument(
        "--save_sampler_masks",
        action="store_true",
        default=False,
    )  # noqa
    parser.add_argument(
        "--use_twolayer_vslnet",
        action="store_true",
        default=False,
    )  # noqa
    parser.add_argument(
        "--twolayer_topk",
        type=int,
        default=1,
    )  # noqa
    parser.add_argument(
        "--pretrained_zeroclip_path",
        type=str,
        default="",
    )  # noqa
    configs = parser.parse_args()
    return configs, parser
