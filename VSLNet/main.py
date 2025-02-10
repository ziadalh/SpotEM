"""Main script to train/test VSLNet models"""

import copy
import os

import numpy as np
import options
import submitit
import torch
import torch.nn as nn
from model.VSLNet import VSLNet, build_optimizer_and_scheduler
from model.VSLNet_2layer import VSLNetTwoLayer
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from utils.data_gen import gen_or_load_dataset
from utils.data_loader import get_test_loader, get_train_loader
from utils.data_util import load_json, load_video_features, save_json
from utils.runner_utils import (
    convert_length_to_mask,
    eval_test,
    filter_checkpoints,
    get_last_checkpoint,
    set_th_config,
)


def main(configs, parser):
    print(f"Running with {configs}", flush=True)

    # set tensorflow configs
    set_th_config(configs.seed)

    # prepare or load dataset
    dataset = gen_or_load_dataset(configs)
    configs.char_size = dataset.get("n_chars", -1)
    configs.word_size = dataset.get("n_words", -1)

    # get train and test loader
    visual_features = load_video_features(
        os.path.join("data", "features", configs.task, configs.fv), configs.max_pos_len
    )
    # If video agnostic, randomize the video features.
    if configs.video_agnostic:
        visual_features = {
            key: np.random.rand(*val.shape) for key, val in visual_features.items()
        }
    train_loader = get_train_loader(
        dataset=dataset["train_set"], video_features=visual_features, configs=configs
    )
    val_loader = (
        None
        if dataset["val_set"] is None
        else get_test_loader(dataset["val_set"], visual_features, configs)
    )
    test_loader = get_test_loader(
        dataset=dataset["test_set"], video_features=visual_features, configs=configs
    )
    configs.num_train_steps = len(train_loader) * configs.epochs
    num_train_batches = len(train_loader)

    # Device configuration
    cuda_str = "cuda" if configs.gpu_idx is None else "cuda:{}".format(configs.gpu_idx)
    device = torch.device(cuda_str if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}")

    # create model dir
    home_dir = os.path.join(
        configs.model_dir,
        "_".join(
            [
                configs.model_name,
                configs.task,
                configs.fv,
                str(configs.max_pos_len),
                configs.predictor,
            ]
        ),
    )
    if configs.suffix is not None:
        home_dir = home_dir + "_" + configs.suffix
    model_dir = os.path.join(home_dir, "model")

    writer = None
    if configs.log_to_tensorboard is not None:
        log_dir = os.path.join(configs.tb_log_dir, configs.log_to_tensorboard)
        os.makedirs(log_dir, exist_ok=True)
        print(f"Writing to tensorboard: {log_dir}")
        writer = SummaryWriter(log_dir=log_dir)

    # train and test
    if configs.mode.lower() == "train":
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if configs.use_feature_sampler:
            eval_efficiency_values = [configs.feature_sampler_efficiency]
            for eff in eval_efficiency_values:
                eff_int = int(eff * 100)
                os.makedirs(
                    os.path.join(model_dir, f"efficiency_{eff_int}"), exist_ok=True
                )

        eval_period = num_train_batches // 2
        save_json(
            vars(configs),
            os.path.join(model_dir, "configs.json"),
            sort_keys=True,
            save_pretty=True,
        )
        # build model
        if configs.use_twolayer_vslnet:
            model = VSLNetTwoLayer(
                configs=configs,
                word_vectors=dataset.get("word_vector", None),
                device=device,
            ).to(device)
        else:
            model = VSLNet(
                configs=configs, word_vectors=dataset.get("word_vector", None)
            ).to(device)
        expert_model = None
        if configs.use_full_distillation:
            assert configs.use_feature_sampler or configs.use_twolayer_vslnet
            expert_configs = copy.deepcopy(configs)
            expert_configs.use_feature_sampler = False
            expert_model = VSLNet(
                configs=expert_configs, word_vectors=dataset.get("word_vector", None)
            )
            expert_ckpt = torch.load(configs.distillation_ckpt_path, map_location="cpu")
            expert_model.load_state_dict(expert_ckpt)
            expert_model.to(device)
            for p in expert_model.parameters():
                p.requires_grad = False
            expert_model.eval()
        optimizer, scheduler = build_optimizer_and_scheduler(model, configs=configs)
        # start training
        best_metric = -1.0
        if configs.use_feature_sampler:
            best_metric = [-1.0]
        score_writer = open(
            os.path.join(model_dir, "eval_results.txt"), mode="w", encoding="utf-8"
        )
        print("start training...", flush=True)
        global_step = 0
        initial_tau = configs.sampler_initial_tau
        final_tau = configs.sampler_final_tau
        delta_tau = (final_tau - initial_tau) / configs.epochs
        for epoch in range(configs.epochs):
            model.train()
            for data in tqdm(
                train_loader,
                total=num_train_batches,
                desc="Epoch %3d / %3d" % (epoch + 1, configs.epochs),
            ):
                global_step += 1
                (
                    _,
                    vfeats,
                    vfeat_lens,
                    word_ids,
                    char_ids,
                    s_labels,
                    e_labels,
                    h_labels,
                ) = data
                # prepare features
                vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
                s_labels, e_labels, h_labels = (
                    s_labels.to(device),
                    e_labels.to(device),
                    h_labels.to(device),
                )
                if isinstance(word_ids, dict):
                    word_ids = {key: val.to(device) for key, val in word_ids.items()}
                    # generate mask
                    query_mask = (
                        (
                            torch.zeros_like(word_ids["input_ids"])
                            != word_ids["input_ids"]
                        )
                        .float()
                        .to(device)
                    )
                else:
                    word_ids, char_ids = word_ids.to(device), char_ids.to(device)
                    # generate mask
                    query_mask = (
                        (torch.zeros_like(word_ids) != word_ids).float().to(device)
                    )
                # generate mask
                video_mask = convert_length_to_mask(vfeat_lens).to(device)

                # compute logits
                curr_tau = initial_tau + epoch * delta_tau
                h_score, start_logits, end_logits, features, sampler_masks = model(
                    word_ids,
                    char_ids,
                    vfeats,
                    video_mask,
                    query_mask,
                    gs_tau=curr_tau,
                    return_features=True,
                )
                # compute loss
                highlight_loss = model.compute_highlight_loss(
                    h_score, h_labels, video_mask
                )
                loc_loss = model.compute_loss(
                    start_logits, end_logits, s_labels, e_labels
                )
                sampler_loss = model.compute_sampler_loss(sampler_masks, video_mask)
                paired_losses_and_weights = [
                    (loc_loss, 1.0),
                    (highlight_loss, configs.highlight_lambda),
                    (sampler_loss, configs.sampler_loss_coef),
                ]
                if expert_model is not None:
                    with torch.no_grad():
                        (
                            e_h_score,
                            e_start_logits,
                            e_end_logits,
                            e_features,
                            _,
                        ) = expert_model(
                            word_ids,
                            char_ids,
                            vfeats,
                            video_mask,
                            query_mask,
                            return_features=True,
                        )
                    distill_h_loss = model.compute_highlight_distillation_loss(
                        h_score, e_h_score, video_mask
                    )
                    distill_l_loss = model.compute_predictor_distillation_loss(
                        start_logits, end_logits, e_start_logits, e_end_logits
                    )
                    distill_f_loss = model.compute_feature_distillation_loss(
                        features, e_features, video_mask
                    )
                    paired_losses_and_weights += [
                        (distill_h_loss, configs.distill_h_coef),
                        (distill_l_loss, configs.distill_l_coef),
                        (distill_f_loss, configs.distill_f_coef),
                    ]

                total_loss = 0.0
                for loss, wt in paired_losses_and_weights:
                    total_loss = total_loss + loss * wt
                # compute and apply gradients
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), configs.clip_norm
                )  # clip gradient
                optimizer.step()
                scheduler.step()
                if writer is not None and global_step % configs.tb_log_freq == 0:
                    writer.add_scalar(
                        "Loss/Total", total_loss.detach().cpu(), global_step
                    )
                    writer.add_scalar("Loss/Loc", loc_loss.detach().cpu(), global_step)
                    writer.add_scalar(
                        "Loss/Highlight", highlight_loss.detach().cpu(), global_step
                    )
                    writer.add_scalar(
                        "Loss/Highlight (*lambda)",
                        (configs.highlight_lambda * highlight_loss.detach().cpu()),
                        global_step,
                    )
                    if model.sampler and type(sampler_loss) is torch.Tensor:
                        writer.add_scalar(
                            "Loss/Sampler", sampler_loss.detach().cpu(), global_step
                        )
                        srate = model.compute_sampling_rate(sampler_masks, video_mask)
                        srate = srate.mean().item()
                        writer.add_scalar("Efficient sampling rate", srate, global_step)
                        writer.add_scalar("GS tau", curr_tau, global_step)
                        if configs.use_full_distillation:
                            writer.add_scalar(
                                "Loss/Distill-Location",
                                distill_l_loss.detach().cpu(),
                                global_step,
                            )
                            writer.add_scalar(
                                "Loss/Distill-Highlight",
                                distill_h_loss.detach().cpu(),
                                global_step,
                            )
                            writer.add_scalar(
                                "Loss/Distill-Features",
                                distill_f_loss.detach().cpu(),
                                global_step,
                            )

                    writer.add_scalar(
                        "LR", optimizer.param_groups[0]["lr"], global_step
                    )

                # evaluate
                if configs.eval_freq > 0:
                    eval_flag = (epoch + 1) % configs.eval_freq == 0
                    eval_flag = eval_flag and global_step % num_train_batches == 0
                else:
                    eval_flag = (
                        global_step % eval_period == 0
                        or global_step % num_train_batches == 0
                    )

                if eval_flag:
                    model.eval()
                    print(
                        f"\nEpoch: {epoch + 1:2d} | Step: {global_step:5d}", flush=True
                    )
                    # Evaluate on val, keep the top 3 checkpoints.
                    if configs.use_feature_sampler:
                        for eff_idx, eff in enumerate(eval_efficiency_values):
                            eff_int = int(eff * 100)
                            model_dir_eff = os.path.join(
                                model_dir, f"efficiency_{eff_int}"
                            )
                            result_save_path = os.path.join(
                                model_dir_eff,
                                f"{configs.model_name}_{epoch}_{global_step}_preds.json",
                            )

                            results, mIoU, (score_str, score_dict) = eval_test(
                                model=model,
                                data_loader=val_loader,
                                device=device,
                                mode="val",
                                epoch=epoch + 1,
                                global_step=global_step,
                                gt_json_path=configs.eval_gt_json,
                                result_save_path=result_save_path,
                            )
                            eff_str = f"========> Evaluating at efficiency {eff_int}"
                            score_writer.write(eff_str + "\n")
                            print(eff_str, flush=True)
                            print(score_str, flush=True)
                            if writer is not None:
                                for name, value in score_dict.items():
                                    kk = name.replace("\n", " ")
                                    writer.add_scalar(
                                        f"Val/efficiency_{eff_int}/{kk}",
                                        value,
                                        global_step,
                                    )
                            eff_str = "===> Estimated efficiency: {:.2f}".format(
                                score_dict["Sampling efficiency"] * 100.0
                            )
                            print(eff_str, flush=True)
                            score_writer.write(eff_str)
                            score_writer.write(score_str)
                            score_writer.flush()

                            # Recall@1, 0.3 IoU overlap --> best metric.
                            if results[0][0] >= best_metric[eff_idx]:
                                save_flag = True
                                if score_dict["Sampling efficiency"] < eff:
                                    save_flag = False
                                if save_flag:
                                    best_metric[eff_idx] = results[0][0]
                                    torch.save(
                                        model.state_dict(),
                                        os.path.join(
                                            model_dir_eff,
                                            "{}_{}.t7".format(
                                                configs.model_name, global_step
                                            ),
                                        ),
                                    )
                                    # only keep the top-3 model checkpoints
                                    filter_checkpoints(
                                        model_dir_eff, suffix="t7", max_to_keep=3
                                    )
                    else:
                        result_save_path = os.path.join(
                            model_dir,
                            f"{configs.model_name}_{epoch}_{global_step}_preds.json",
                        )
                        # Evaluate on val, keep the top 3 checkpoints.
                        results, mIoU, (score_str, score_dict) = eval_test(
                            model=model,
                            data_loader=val_loader,
                            device=device,
                            mode="val",
                            epoch=epoch + 1,
                            global_step=global_step,
                            gt_json_path=configs.eval_gt_json,
                            result_save_path=result_save_path,
                        )
                        print(score_str, flush=True)
                        if writer is not None:
                            for name, value in score_dict.items():
                                kk = name.replace("\n", " ")
                                writer.add_scalar(f"Val/{kk}", value, global_step)

                        eff_str = "===> Estimated efficiency: {:.2f}".format(
                            score_dict["Sampling efficiency"] * 100.0
                        )
                        print(eff_str, flush=True)
                        score_writer.write(score_str)
                        score_writer.flush()
                        # Recall@1, 0.3 IoU overlap --> best metric.
                        if results[0][0] >= best_metric:
                            save_flag = True
                            min_eff = configs.feature_sampler_efficiency
                            if (
                                configs.use_feature_sampler
                                and score_dict["Sampling efficiency"] < min_eff
                            ):
                                save_flag = False

                            if save_flag:
                                best_metric = results[0][0]
                                torch.save(
                                    model.state_dict(),
                                    os.path.join(
                                        model_dir,
                                        "{}_{}.t7".format(
                                            configs.model_name, global_step
                                        ),
                                    ),
                                )
                                # only keep the top-3 model checkpoints
                                filter_checkpoints(
                                    model_dir, suffix="t7", max_to_keep=3
                                )
                    model.train()

        score_writer.close()

    elif configs.mode.lower() == "val":
        if not os.path.exists(model_dir):
            raise ValueError("No pre-trained weights exist")
        # load previous configs
        pre_configs = load_json(os.path.join(model_dir, "configs.json"))
        parser.set_defaults(**pre_configs)
        configs = parser.parse_args()
        # build model
        model = VSLNet(
            configs=configs, word_vectors=dataset.get("word_vector", None)
        ).to(device)

        # get last checkpoint file
        if configs.use_feature_sampler:
            eff_int = int(configs.feature_sampler_efficiency * 100)
            model_dir_eff = os.path.join(model_dir, f"efficiency_{eff_int}")
            filename = get_last_checkpoint(model_dir_eff, suffix="t7")
        else:
            filename = get_last_checkpoint(model_dir, suffix="t7")
        model.load_state_dict(torch.load(filename))
        model.eval()
        result_save_path = filename.replace(".t7", "_val_result.json")
        results, mIoU, score_str = eval_test(
            model=model,
            data_loader=val_loader,
            device=device,
            mode="val",
            gt_json_path=configs.eval_gt_json,
            result_save_path=result_save_path,
            save_sampler_masks=configs.save_sampler_masks,
        )
        print(score_str, flush=True)

    elif configs.mode.lower() == "test":
        if not os.path.exists(model_dir):
            raise ValueError("No pre-trained weights exist")
        # load previous configs
        pre_configs = load_json(os.path.join(model_dir, "configs.json"))
        parser.set_defaults(**pre_configs)
        configs = parser.parse_args()
        # build model
        model = VSLNet(
            configs=configs, word_vectors=dataset.get("word_vector", None)
        ).to(device)

        # get last checkpoint file
        filename = get_last_checkpoint(model_dir, suffix="t7")
        model.load_state_dict(torch.load(filename))
        model.eval()
        result_save_path = filename.replace(".t7", "_test_result.json")
        results, mIoU, score_str = eval_test(
            model=model,
            data_loader=test_loader,
            device=device,
            mode="test",
            result_save_path=result_save_path,
        )
        print(score_str, flush=True)


def create_executor(configs):
    executor = submitit.AutoExecutor(folder=configs.slurm_log_folder)

    executor.update_parameters(
        timeout_min=configs.slurm_timeout_min,
        constraint=configs.slurm_constraint,
        slurm_partition=configs.slurm_partition,
        gpus_per_node=configs.slurm_gpus,
        cpus_per_task=configs.slurm_cpus,
    )
    return executor


if __name__ == "__main__":
    configs, parser = options.read_command_line()
    if not configs.slurm:
        main(configs, parser)
    else:
        executor = create_executor(configs)

        job = executor.submit(main, configs, parser)
        print("job=", job.job_id)

        # wait for it
        if configs.slurm_wait:
            job.result()
