import copy
import logging
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as distrib
import torch.nn as nn
from easydict import EasyDict as edict
from ms_cm.configs import BaseOptions
from ms_cm.inference_ego4d_slowfast import eval_epoch, setup_model
from ms_cm.sw_vs_ego4d_dataset import Ego4d_collate, Ego4d_dataset, prepare_batch_inputs
from ms_cm.vslnet_utils.data_gen_ego4d import gen_or_load_dataset
from ms_cm.vslnet_utils.distributed_training import ddp_setup, get_distrib_size
from ms_cm.vslnet_utils.runner_utils import filter_checkpoints
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.basic_utils import AverageMeter, dict_to_markdown
from utils.model_utils import count_parameters

torch.set_num_threads(1)


def extract_index(start_logits, end_logits):
    start_prob = nn.Softmax(dim=1)(start_logits)
    end_prob = nn.Softmax(dim=1)(end_logits)
    outer = torch.matmul(start_prob.unsqueeze(dim=2), end_prob.unsqueeze(dim=1))
    outer = torch.triu(outer, diagonal=0)

    # Get top 5 start and end indices.
    batch_size, height, width = outer.shape
    outer_flat = outer.view(batch_size, -1)
    _, flat_indices = outer_flat.topk(5, dim=-1)
    start_indices = flat_indices // width
    end_indices = flat_indices % width
    return start_indices, end_indices


def index_to_time(start_index, end_index, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
    e_times = (
        np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
    )
    start_time = s_times[start_index]
    end_time = e_times[end_index]
    return start_time, end_time


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model,
    criterion,
    train_loader,
    optimizer,
    opt,
    epoch_i,
    global_step,
    tb_writer,
    logger,
    is_distributed,
    local_rank,
    world_rank,
    world_size,
    gt_json_path="../../Datasets/Ego4d/ego4d_annotation/nlq_train.json",
    expert_model=None,
):
    if world_rank == 0:
        logger.info(f"[Epoch {epoch_i + 1}]")
    model.train()
    criterion.train()

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    if world_rank == 0:
        iterator = tqdm(
            train_loader,
            desc=f"Epoch: {epoch_i}",
            total=num_training_examples,
        )
    else:
        iterator = train_loader

    initial_tau = opt.sampler_initial_tau
    final_tau = opt.sampler_final_tau
    delta_tau = (final_tau - initial_tau) / opt.n_epoch
    curr_tau = initial_tau + epoch_i * delta_tau

    sampler = model.module.sampler if is_distributed else model.sampler

    for _, batch in enumerate(iterator):
        global_step += 1
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        timer_start = time.time()
        model_inputs, targets = prepare_batch_inputs(
            batch[1],
            opt.device,
            non_blocking=opt.pin_memory,
            cur_scale=batch[2],
        )
        time_meters["prepare_inputs_time"].update(time.time() - timer_start)

        outputs = model(**model_inputs, gs_tau=curr_tau)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        if sampler and type(outputs["sampler_masks"]) is torch.Tensor:
            s_loss = sampler.compute_loss(
                outputs["sampler_masks"], outputs["video_mask"]
            )
            loss_dict["sampler_loss"] = s_loss
            weight_dict["sampler_loss"] = opt.sampler_loss_coef
            if expert_model is not None:
                with torch.no_grad():
                    expert_outputs = expert_model(**model_inputs)
                d_sal = expert_model.compute_saliency_distillation_loss(
                    outputs["saliency_scores"], expert_outputs["saliency_scores"]
                )
                loss_dict["distill-saliency"] = d_sal
                weight_dict["distill-saliency"] = opt.distill_s_coef
                d_npm = expert_model.compute_npm_distillation_loss(
                    outputs["NPM_scores"],
                    expert_outputs["NPM_scores"],
                    outputs["scale_mask"],
                )
                loss_dict["distill-npm"] = d_npm
                weight_dict["distill-npm"] = opt.distill_n_coef
                d_hig = expert_model.compute_highlight_distillation_loss(
                    outputs["highlight_scores"],
                    expert_outputs["highlight_scores"],
                    outputs["video_mask"],
                )
                loss_dict["distill-highlight"] = d_hig
                weight_dict["distill-highlight"] = opt.distill_h_coef
                d_loc = expert_model.compute_predictor_distillation_loss(
                    outputs["start_logits"],
                    outputs["end_logits"],
                    expert_outputs["start_logits"],
                    expert_outputs["end_logits"],
                )
                loss_dict["distill-location"] = d_loc
                weight_dict["distill-location"] = opt.distill_l_coef
                d_fea = expert_model.compute_feature_distillation_loss(
                    outputs["video_memory"],
                    expert_outputs["video_memory"],
                    outputs["video_mask"],
                )
                loss_dict["distill-features"] = d_fea
                weight_dict["distill-features"] = opt.distill_f_coef

        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        time_meters["model_forward_time"].update(time.time() - timer_start)

        timer_start = time.time()
        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = float(losses)  # for logging only
        if sampler and type(outputs["sampler_masks"]) is torch.Tensor:
            s_rate = sampler.compute_sampling_rate(
                outputs["sampler_masks"], outputs["video_mask"]
            )
            loss_dict["efficient_sampling_rate"] = (
                s_rate.mean().item()
            )  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(
                float(v) * weight_dict[k] if k in weight_dict else float(v)
            )

        timer_dataloading = time.time()

        if global_step % opt.period == 0:
            loss_key_vals = [(k, v.avg) for k, v in loss_meters.items()]
            loss_keys, loss_vals = zip(*loss_key_vals)
            if is_distributed:
                # Synchronize stats across workers
                stats_to_sync = torch.Tensor(loss_vals).to(opt.device)
                distrib.all_reduce(stats_to_sync)
                stats_to_sync = stats_to_sync / world_size
                loss_vals = stats_to_sync.cpu().numpy().tolist()
            if world_rank == 0:
                # print/add logs
                tb_writer.add_scalar(
                    "Train/lr", float(optimizer.param_groups[0]["lr"]), global_step
                )
                tb_writer.add_scalar("Train/epoch", epoch_i, global_step)
                for k, v in zip(loss_keys, loss_vals):
                    tb_writer.add_scalar("Train/{}".format(k), v, global_step)
                to_write = opt.train_log_txt_formatter.format(
                    time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                    epoch=epoch_i + 1,
                    loss_str=" ".join(
                        ["{} {:.4f}".format(k, v) for k, v in zip(loss_keys, loss_vals)]
                    ),
                )
                with open(opt.train_log_filepath, "a") as f:
                    f.write(to_write)

    if world_rank == 0:
        logger.info("Epoch time stats:")
        for name, meter in time_meters.items():
            d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
            logger.info(f"{name} ==> {d}")

    return global_step


def train(
    model,
    criterion,
    optimizer,
    lr_scheduler,
    train_dataset,
    val_dataset,
    opt,
    logger,
    is_distributed,
    local_rank,
    world_rank,
    world_size,
    expert_model=None,
):
    best_metric = -1.0
    if opt.use_feature_sampler:
        best_metric = [-1.0]

    score_writer = None
    tb_writer = None
    if world_rank == 0:
        score_writer = open(
            os.path.join(opt.results_dir, "eval_results.txt"),
            mode="w",
            encoding="utf-8",
        )
        tb_writer = SummaryWriter(opt.tensorboard_log_dir)
        tb_writer.add_text(
            "hyperparameters", dict_to_markdown(vars(opt), max_str_len=None)
        )

    if opt.use_feature_sampler:
        eval_efficiency_values = [opt.feature_sampler_efficiency]
        if world_rank == 0:
            for eff in eval_efficiency_values:
                eff_int = int(eff * 100)
                os.makedirs(
                    os.path.join(opt.results_dir, f"efficiency_{eff_int}"),
                    exist_ok=True,
                )

    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_sampler = None
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=Ego4d_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=opt.pin_memory,
    )

    # Avoid distributed evaluation
    eval_loader = DataLoader(
        val_dataset,
        collate_fn=Ego4d_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory,
    )
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    global_step = 0
    for epoch_i in range(start_epoch, opt.n_epoch):
        if is_distributed:
            train_sampler.set_epoch(epoch_i)

        if epoch_i > -1:
            global_step = train_epoch(
                model,
                criterion,
                train_loader,
                optimizer,
                opt,
                epoch_i,
                global_step,
                tb_writer,
                logger,
                is_distributed,
                local_rank,
                world_rank,
                world_size,
                expert_model=expert_model,
            )
            if opt.optim_name == "AdamW":
                lr_scheduler.step()

        eval_epoch_interval = opt.eval_epoch_interval
        # Evaluate only within rank 0
        if world_rank == 0 and (epoch_i + 1) % eval_epoch_interval == 0:
            print(f"\nEpoch: {epoch_i + 1:2d} | Step: {global_step} |", flush=True)
            if opt.use_feature_sampler:
                model.eval()
                model_eval = model.module if is_distributed else model
                for eff_idx, eff in enumerate(eval_efficiency_values):
                    eff_int = int(eff * 100)
                    results_dir_eff = os.path.join(
                        opt.results_dir, f"efficiency_{eff_int}"
                    )
                    result_save_path = os.path.join(
                        results_dir_eff,
                        f"{epoch_i}_preds.json",
                    )

                    with torch.no_grad():
                        results, mIoU, (score_str, score_dict) = eval_epoch(
                            model_eval,
                            eval_loader,
                            opt,
                            result_save_path,
                            opt.eval_gt_json,
                            epoch_i,
                            tb_writer,
                            return_results_dict=True,
                        )
                    eff_str = f"========> Evaluating at efficiency {eff_int}"
                    for name, value in score_dict.items():
                        tb_writer.add_scalar(
                            f"Val/efficiency_{eff_int}/{name}",
                            value,
                            global_step,
                        )
                    print(eff_str)
                    print(score_str, flush=True)
                    score_writer.write(eff_str + "\n")
                    eff_str = "===> Estimated efficiency: {:.2f}".format(
                        score_dict["Sampling efficiency"] * 100.0
                    )
                    print(eff_str, flush=True)
                    score_writer.write(eff_str + "\n")
                    score_writer.write(
                        f"\nEpoch: {epoch_i + 1} | Step: {global_step}\n"
                    )
                    score_writer.write(score_str)
                    score_writer.flush()

                    if opt.optim_name == "AdamW":
                        checkpoint = {
                            "model": model.module.state_dict()
                            if is_distributed
                            else model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch_i,
                            "opt": opt,
                        }
                    elif opt.optim_name == "BertAdam":
                        checkpoint = {
                            "model": model.module.state_dict()
                            if is_distributed
                            else model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch_i,
                            "opt": opt,
                        }
                    print("_{:0>4d}.ckpt".format(epoch_i))
                    # Recall@1, 0.3 IoU overlap --> best metric.
                    if results[0][0] >= best_metric[eff_idx]:
                        save_flag = True
                        if score_dict["Sampling efficiency"] < eff:
                            save_flag = False

                        if save_flag:
                            best_metric[eff_idx] = results[0][0]
                            torch.save(
                                checkpoint,
                                os.path.join(
                                    results_dir_eff,
                                    "model_{}.t7".format(global_step),
                                ),
                            )
                            # only keep top-3 model checkpoints
                            filter_checkpoints(
                                results_dir_eff, suffix="t7", max_to_keep=3
                            )
            else:
                result_save_path = os.path.join(
                    opt.results_dir,
                    f"{epoch_i}_preds.json",
                )
                model.eval()
                model_eval = model.module if is_distributed else model
                with torch.no_grad():
                    results, mIoU, (score_str, score_dict) = eval_epoch(
                        model_eval,
                        eval_loader,
                        opt,
                        result_save_path,
                        opt.eval_gt_json,
                        epoch_i,
                        tb_writer,
                        return_results_dict=True,
                    )
                for name, value in score_dict.items():
                    tb_writer.add_scalar(f"Val/{name}", value, global_step)

                print(score_str, flush=True)
                score_writer.write(score_str)
                score_writer.flush()

                if opt.optim_name == "AdamW":
                    checkpoint = {
                        "model": model.module.state_dict()
                        if is_distributed
                        else model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch_i,
                        "opt": opt,
                    }
                elif opt.optim_name == "BertAdam":
                    checkpoint = {
                        "model": model.module.state_dict()
                        if is_distributed
                        else model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch_i,
                        "opt": opt,
                    }
                print("_{:0>4d}.ckpt".format(epoch_i))
                # Recall@1, 0.3 IoU overlap --> best metric.
                if results[0][0] >= best_metric:
                    save_flag = True
                    min_eff = opt.feature_sampler_efficiency
                    if (
                        opt.use_feature_sampler
                        and score_dict["Sampling efficiency"] < min_eff
                    ):
                        save_flag = False

                    if save_flag:
                        best_metric = results[0][0]
                        torch.save(
                            checkpoint,
                            os.path.join(
                                opt.results_dir,
                                "model_{}.t7".format(global_step),
                            ),
                        )
                        # only keep top-3 model checkpoints
                        filter_checkpoints(opt.results_dir, suffix="t7", max_to_keep=3)

    if world_rank == 0:
        tb_writer.close()
        score_writer.close()


def start_training(is_distributed, local_rank, world_rank, world_size):
    base_opt = BaseOptions()
    opt = base_opt.parse()
    if world_rank == 0:
        base_opt.display_save(opt)

    if world_rank == 0:
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )
        logger.info("Setup config, data and model...")
    else:
        logger = None

    if is_distributed:
        assert opt.bsz % world_size == 0
        opt.bsz = opt.bsz // world_size
        if world_rank == 0:
            logger.info(f"=====> Setting batch_size to {opt.bsz}")
    opt.device = torch.device(f"cuda:{local_rank}")

    set_seed(opt.seed + world_rank)
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()

    dataset_config = dict(
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        multiscale_list=opt.numscale_list,
        use_sw=opt.use_sw,
        sw_len_ratio=opt.sw_len_ratio,
        use_vs=opt.use_vs,
        vs_prob=opt.vs_prob,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        txt_drop_ratio=opt.txt_drop_ratio,
        nar_rand_exp_factor=opt.nar_rand_window_expansion_factor,
        nar_rand_translate=opt.nar_rand_window_translate,
    )
    vslnet_datasetconfigs = edict(
        {
            "task": opt.vsldataset_task,
            "fv": opt.vsldataset_fv,
            "max_pos_len": opt.max_v_l,
            "num_workers": opt.vsldataset_num_workers,
            "vslnet_datapath": opt.vslnet_datapath,
            "save_dir": opt.vslnet_dataset_save_dir,
            "thres_in_train": opt.vslnet_thres_in_train,
        }
    )
    # preparing ego4d dataset by using vslnet configuration
    dataset = gen_or_load_dataset(vslnet_datasetconfigs)

    dataset_config["dataset"] = dataset["train_set"]
    dataset_config["mode"] = "train"
    train_dataset = Ego4d_dataset(**dataset_config)

    dataset_config["dataset"] = dataset["val_set"]
    dataset_config["mode"] = "eval"
    dataset_config["txt_drop_ratio"] = 0
    dataset_config["nar_rand_exp_factor"] = -1.0
    dataset_config["nar_rand_translate"] = False
    eval_dataset = Ego4d_dataset(**dataset_config)

    model, criterion, optimizer, lr_scheduler = setup_model(
        opt, is_distributed, local_rank, world_rank
    )
    if opt.use_full_distillation:
        assert opt.use_feature_sampler
        expert_opt = copy.deepcopy(opt)
        expert_opt.use_feature_sampler = False
        expert_model, _, _, _ = setup_model(expert_opt, False, 0, 0)
        ckpt = torch.load(opt.distillation_ckpt_path, map_location="cpu")["model"]
        expert_model.load_state_dict(ckpt)
        for p in expert_model.parameters():
            p.requires_grad = False
        expert_model.to(opt.device)
        expert_model.eval()
    else:
        expert_model = None

    if world_rank == 0:
        logger.info(f"Model {model}")
        count_parameters(model)
        logger.info("Start Training...")
    train(
        model,
        criterion,
        optimizer,
        lr_scheduler,
        train_dataset,
        eval_dataset,
        opt,
        logger,
        is_distributed,
        local_rank,
        world_rank,
        world_size,
        expert_model=expert_model,
    )


if __name__ == "__main__":
    is_distributed = get_distrib_size()[2] > 1
    local_rank, world_rank, world_size = 0, 0, 1
    if is_distributed:
        local_rank, world_rank, world_size = ddp_setup()
    # Setup DDP
    start_training(is_distributed, local_rank, world_rank, world_size)
