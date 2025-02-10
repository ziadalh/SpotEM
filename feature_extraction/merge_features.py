import argparse
import glob
import json
import multiprocessing as mp
import os

import torch
import tqdm


def merge_feats(video_name, *src_dirs, tgt_dir="./"):
    feats = []
    tgt_path = os.path.join(tgt_dir, video_name + ".pt")
    if os.path.isfile(tgt_path):
        try:
            shp = torch.load(tgt_path).shape[0]
            return (video_name, shp)
        except Exception as e:
            print(f"Unable to load features from {tgt_path}: {e}")

    src_paths = [os.path.join(d, video_name + ".pt") for d in src_dirs]
    for path in src_paths:
        feats.append(torch.load(path))
    min_len = min([f.shape[0] for f in feats])
    feats = [f[:min_len] for f in feats]
    mfeat = torch.cat(feats, dim=1)

    torch.save(mfeat, tgt_path)
    return (video_name, mfeat.shape[0])


def merge_video_feats(inputs):
    args, video_name = inputs
    return merge_feats(video_name, *args.src_dirs, tgt_dir=args.tgt_dir)


def main(args):
    src_dirs = args.src_dirs
    # Find common set of videos across features
    common_video_names = None
    for d in src_dirs:
        paths = sorted(glob.glob(os.path.join(d, "*.pt")))
        video_names = set([os.path.basename(p)[: -len(".pt")] for p in paths])
        if common_video_names is None:
            common_video_names = video_names
        else:
            common_video_names &= video_names
    common_video_names = list(common_video_names)
    print(f"# common videos: {len(common_video_names)}")

    os.makedirs(args.tgt_dir, exist_ok=True)

    inputs = [(args, v) for v in common_video_names]

    feature_shapes = {}
    with mp.Pool(32, maxtasksperchild=1) as pool, tqdm.tqdm(total=len(inputs)) as pbar:
        for out in pool.imap_unordered(merge_video_feats, inputs):
            pbar.update()
            feature_shapes[out[0]] = out[1]
    with open(os.path.join(args.tgt_dir, "feature_shapes.json"), "w") as fp:
        json.dump(feature_shapes, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dirs", type=str, required=True, nargs="+")
    parser.add_argument("--tgt_dir", type=str, required=True)

    args = parser.parse_args()

    main(args)
