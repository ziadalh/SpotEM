import glob
import os

import decord
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo

decord.bridge.set_bridge("torch")


def sample_frames_clips(start, end, vlen, acc_samples):
    start = max(0, start)
    end = min(vlen, end)

    intervals = np.linspace(start=start, stop=end, num=int(acc_samples) + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    return frame_idxs


def get_video_transform(
    input_res=224,
    center_crop=256,
    norm_mean=(0.485, 0.456, 0.406),
    norm_std=(0.229, 0.224, 0.225),
):
    normalize = NormalizeVideo(mean=norm_mean, std=norm_std)
    transform = transforms.Compose(
        [
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
        ]
    )
    return transform


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root, video_fmt):
        super().__init__()
        self.root = root
        self.fps = 30
        self.fps_to_read = 1.87
        self.video_fmt = video_fmt
        self.video_paths = sorted(glob.glob(os.path.join(root, f"*.{video_fmt}")))
        self.transforms = get_video_transform()

    def __len__(self):
        return len(self.video_paths)

    def read_frames_decord(self, video_reader, start, end, num_frames):
        vlen = len(video_reader)
        frame_idxs = sample_frames_clips(start, end, vlen, num_frames + 1)
        video_reader.skip_frames(1)
        frames = video_reader.get_batch(frame_idxs)

        frames = frames.float() / 255
        frames = frames.permute(0, 3, 1, 2)
        return frames, frame_idxs

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_reader = decord.VideoReader(video_path, num_threads=1)
        vlen = len(video_reader)
        start = 0
        end = vlen - 1
        num_frames = vlen / self.fps * self.fps_to_read
        frames, _ = self.read_frames_decord(
            video_reader, start, end, num_frames
        )  # (L, C, H, W)

        frames = frames.transpose(0, 1)  # [L, C, H, W] ---> [C, L, H, W]
        frames = self.transforms(frames)
        frames = frames.transpose(0, 1)  # recover

        return frames

    def get_video_name(self, idx):
        base_name = os.path.basename(self.video_paths[idx])
        assert base_name.endswith(f".{self.video_fmt}")
        return base_name[: -len(f".{self.video_fmt}")]
