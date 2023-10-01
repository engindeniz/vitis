import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from constants import MSVD, ACTIVITYNET, TGIF, MSRVTT

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Feature merger")

    parser.add_argument("--folder", type=str, required=True, help="folder of features")
    parser.add_argument(
        "--output_path", type=str, required=True, help="output path for features"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset",
        required=True,
        choices=[
            MSRVTT, MSVD, ACTIVITYNET, TGIF
        ],
    )
    parser.add_argument(
        "--pad",
        type=int,
        help="set as diff of 0 to trunc and pad up to a certain nb of seconds",
        default=0,
    )

    args = parser.parse_args()
    files = os.listdir(args.folder)
    files = [x for x in files if x[-4:] == ".npy"]

    # Get mapping from feature file name to dataset video_id
    if args.dataset == MSRVTT:
        mapping = {x: int(x.split(".")[0][5:]) for x in files}
    elif args.dataset in [MSVD]:
        mapping = {x: int(x[:-4]) for x in files}
    elif args.dataset in [ACTIVITYNET, TGIF]:
        mapping = {x: x[:-4] for x in files}
    else:
        raise NotImplementedError

    features = {}
    for i in tqdm(range(len(files))):
        x = files[i]
        feat = torch.from_numpy(np.load(os.path.join(args.folder, x)))
        if args.pad and len(feat) < args.pad:
            feat = torch.cat([feat, torch.zeros(args.pad - len(feat), feat.shape[1])])
        elif args.pad:
            feat = feat[: args.pad]
        features[mapping[x]] = feat.half()

    torch.save(features, args.output_path)
