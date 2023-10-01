import json
import h5py
import numpy as np
import pandas as pd
import torch
import torch as th
from torch.utils.data import Dataset

from constants import VITIS
from model.model_utils import get_tokenizer
from util.misc import get_video_mask, mask_tokens_text_dataloader


class VideoText_Dataset(Dataset):
    def __init__(self, args, csv_path, features_path, features_dim=768, webvid_group_image_ids_path=None):
        self.args = args
        self.data = pd.read_csv(csv_path)
        self.data.dropna(inplace=True)
        self.features_path = features_path
        self.webvid_group_image_ids_path = webvid_group_image_ids_path
        self.max_feats = self.args.num_frames
        self.features_dim = features_dim
        self.hf = None
        with open(self.webvid_group_image_ids_path) as f:
            self.group_image_ids = json.load(f)
        self.tokenizer = get_tokenizer(args)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.hf == None:
            self.hf = h5py.File(self.features_path, 'r')

        text = self.data["name"].values[idx]
        video_id = self.data["videoid"].values[idx]

        group_id = self.group_image_ids[str(video_id)]
        video = th.from_numpy(np.array(self.hf[group_id][str(video_id)]))
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = th.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = th.cat(
                [video, th.zeros(self.max_feats - video_len, self.features_dim)], 0
            )
        else:
            video_len = self.max_feats

        video_size = calculate_video_size_webvid(self.args, video)
        video_mask = get_video_mask(video_len, video_size)
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.args.max_tokens,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        inputs, labels = mask_tokens_text_dataloader(
            encoded["input_ids"], self.tokenizer, mlm_probability=self.args.mlm_prob
        )
        attention_mask = encoded["attention_mask"]
        return {"video": video, "video_mask": video_mask, "inputs": inputs, "attention_mask": attention_mask,
                "labels": labels}


def calculate_video_size_webvid(args, video):
    video_size = args.mapping_prompt_num_tokens
    return video_size


def videotext_collate_fn_w_tokenizer(batch):
    padding_token_id = 0
    bs = len(batch)
    video = th.stack([batch[i]["video"] for i in range(bs)])
    video_mask = th.cat([batch[i]["video_mask"] for i in range(bs)])
    max_length_input = max([a['inputs'].size(1) for a in batch])

    inputs = []
    attention_mask = []
    labels = []
    for i in range(bs):
        padded_inputs = torch.zeros((1, max_length_input), dtype=batch[i]['inputs'].dtype)
        padded_attention_mask = torch.zeros((1, max_length_input), dtype=batch[i]['attention_mask'].dtype)
        padded_labels = torch.full((1, max_length_input), -100, dtype=batch[i]['labels'].dtype)

        padded_inputs[:, :batch[i]['inputs'].size(1)] = batch[i]['inputs']
        padded_attention_mask[:, :batch[i]['attention_mask'].size(1)] = batch[i]['attention_mask']
        padded_labels[:, :batch[i]['labels'].size(1)] = batch[i]['labels']
        inputs.append(padded_inputs)
        attention_mask.append(padded_attention_mask)
        labels.append(padded_labels)
    inputs = th.cat(inputs)
    attention_mask = th.cat(attention_mask)
    labels = th.cat(labels)
    return {
        "video": video,
        "video_mask": video_mask,
        "inputs": inputs,
        "attention_mask": attention_mask,
        "labels": labels
    }


def build_videotext_dataset(split, args):
    if split == "train":
        csv_path = args.webvid_train_csv_path
        features_path = args.webvid_train_features_path
        webvid_group_image_ids_path = args.webvid_train_group_image_ids_path
    elif split == "val":
        csv_path = args.webvid_val_csv_path
        features_path = args.webvid_val_features_path
        webvid_group_image_ids_path = args.webvid_val_group_image_ids_path
    else:
        raise NotImplementedError
    if args.model_name == VITIS:
        return VideoText_Dataset(args=args, csv_path=csv_path, features_path=features_path,
                                 features_dim=args.features_dim,
                                 webvid_group_image_ids_path=webvid_group_image_ids_path)

    else:
        raise NotImplementedError
