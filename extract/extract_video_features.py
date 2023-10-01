import argparse
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.modeling_utils import load_state_dict

from args import get_args_parser
from extract.video_loader import VideoLoader
from model.clip.configuration_clip import CLIPVisionConfig
from model.clip.modelling_clip import CLIPVisionModelCustom
from utils import initialize_seeds

SEED = 3


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = torch.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor


class Preprocessing(object):
    def __init__(self):
        self.norm = Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    def __call__(self, tensor):
        tensor = tensor / 255.0
        tensor = self.norm(tensor)
        return tensor


def seed_worker(worker_id):
    """
    Dataloader seed
    Borrowed from https://pytorch.org/docs/stable/notes/randomness.html """
    np.random.seed(SEED)
    random.seed(SEED)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()

    initialize_seeds(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)

    dataset = VideoLoader(
        args.feature_extraction_csv,
        args.feature_extraction_video_main_path,
        args.feature_extraction_features_main_path,
        args.dataset_name
    )

    feature_dim = 768
    n_dataset = len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=args.num_workers,
        sampler=None)

    # Load Video Encoder
    visual_config = CLIPVisionConfig.from_pretrained(
        pretrained_model_name_or_path='./checkpoints/openai-clip-vit-large-patch14')
    visual_encoder_name = 'openai-clip-vit-large-patch14'
    model = CLIPVisionModelCustom(visual_config, args)
    visual_state_dict = load_state_dict(os.path.join('./checkpoints/', visual_encoder_name, 'pytorch_model.bin'))
    model.load_state_dict(visual_state_dict, strict=False)

    model.eval()
    model = model.cuda()

    preprocess = Preprocessing()

    with torch.no_grad():
        for k, data in enumerate(loader):
            input_file = data["input"][0]
            output_file = data["output"][0]
            if len(data["video"].shape) > 3:
                print(
                    "Computing features of video {}/{}: {}".format(
                        k + 1, n_dataset, input_file
                    )
                )
                video = data["video"].squeeze()
                if len(video.shape) == 4:
                    video = preprocess(video)
                    n_chunk = len(video)
                    features = torch.cuda.FloatTensor(n_chunk, feature_dim).fill_(0)
                    n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                    for i in tqdm(range(n_iter)):
                        min_ind = i * args.batch_size
                        max_ind = (i + 1) * args.batch_size
                        video_batch = video[min_ind:max_ind].cuda()
                        batch_features = model(video_batch)
                        features[min_ind:max_ind] = batch_features
                    features = features.cpu().numpy()
                    np.save(output_file, features)
            else:
                print("ERROR: shape of the video: ", len(data["video"].shape))
                print("Video {} already processed or there is an error.".format(input_file))
