# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import argparse
import os
from queue import Empty as QueueEmpty

import torch
import tqdm
from dataset import VideoDataset
from model import EfficientNet_b0
from torch import multiprocessing as mp


def generate_effnet_features(args, net, idx, device_id):
    device = torch.device(f"cuda:{device_id}")

    dataset = VideoDataset(args.videos_root, args.video_format)
    v_id = dataset.get_video_name(idx)

    save_path = f"{args.save_dir}/{v_id}.pt"

    found_ckpt = False
    if os.path.isfile(save_path):
        try:
            _ = torch.load(save_path)
            found_ckpt = True
        except Exception:
            found_ckpt = False
    if found_ckpt:
        return None

    frames = dataset[idx]  # (T, C, H, W)
    bs = args.batch_size
    feats = []
    for i in range(0, len(frames), bs):
        frames_ = frames[i : (i + bs)]
        with torch.no_grad():
            out = net(frames_.to(device)).cpu()
        feats.append(out)
    feats = torch.cat(feats, 0)
    torch.save(feats, save_path)


class Task:
    def __init__(self, args, idx):
        super().__init__()
        self.args = args
        self.idx = idx

    def run(self, net, device_id):
        return generate_effnet_features(self.args, net, self.idx, device_id)


class WorkerWithDevice(mp.Process):
    def __init__(
        self,
        args,
        task_queue: mp.Queue,
        results_queue: mp.Queue,
        worker_id: int,
        device_id: str,
    ):
        self.args = args
        self.device_id = device_id
        self.worker_id = worker_id
        super().__init__(target=self.work, args=(task_queue, results_queue))

    def work(self, task_queue, results_queue):
        net = EfficientNet_b0(
            dim=self.args.dim, pretrained=self.args.pretrained_ckpt == ""
        )
        if self.args.pretrained_ckpt != "":
            ckpt = torch.load(self.args.pretrained_ckpt)
            net.load_state_dict(ckpt["backbone"])
        device = torch.device(f"cuda:{self.device_id}")
        net.eval().to(device)

        while True:
            try:
                task = task_queue.get()
            except QueueEmpty:
                break
            task.run(net, self.device_id)
            results_queue.put(task.idx)
            del task

        del net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_root", required=True, type=str)
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--n_processes_per_gpu", default=4, type=int)
    parser.add_argument("--dim", default=1280, type=int)
    parser.add_argument("--pretrained_ckpt", default="", type=str)
    parser.add_argument("--video_format", default="mp4", type=str)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = VideoDataset(args.videos_root, args.video_format)
    n_devices = torch.cuda.device_count()

    mp.set_start_method("forkserver")

    task_queue = mp.Queue()
    for i in range(len(dataset)):
        task = Task(args, i)
        task_queue.put(task)

    results_queue = mp.Queue()

    pbar = tqdm.tqdm(
        desc="Computing EfficientNet features",
        position=0,
        total=len(dataset),
    )

    workers = [
        WorkerWithDevice(args, task_queue, results_queue, i, i % n_devices)
        for i in range(args.n_processes_per_gpu * n_devices)
    ]
    # Start workers
    for worker in workers:
        worker.start()
    # Update progress bar
    n_completed = 0
    while n_completed < len(dataset):
        _ = results_queue.get()
        n_completed += 1
        pbar.update()
    # Wait for workers to finish
    for worker in workers:
        worker.join()
