import os
import time

import ffmpeg
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from constants import WEBVID, TGIF, MSVD, ACTIVITYNET, MSRVTT


class VideoLoader(Dataset):
    """Pytorch video loader."""

    def __init__(
            self,
            csv,
            video_main_path,
            features_main_path,
            dataset_name
    ):

        self.csv = pd.read_csv(csv)
        self.video_main_path = video_main_path
        self.features_main_path = features_main_path
        self.dataset_name = dataset_name
        self.centercrop = True
        self.size = 224
        self.framerate = 1
        if self.dataset_name == WEBVID:
            # Drop nan values
            self.csv.dropna(inplace=True)
            self.csv = self.csv.sample(frac=1, random_state=int(str(time.time()).split(".")[1])).reset_index(drop=True)

    def __len__(self):
        return len(self.csv)

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        num, denum = video_stream["avg_frame_rate"].split("/")
        frame_rate = int(num) / int(denum)
        return height, width, frame_rate

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def _get_video_clip(self, video_id, video_path):

        if os.path.isfile(video_path):
            print("Decoding video: {}".format(video_id))
            try:
                h, w, fr = self._get_video_dim(video_path)
            except:
                video = torch.zeros(1)
                return video

            if fr < 1:
                video = torch.zeros(1)
                return video

            height, width = self._get_output_dim(h, w)
            try:
                cmd = (
                    ffmpeg.input(video_path)
                    .filter("fps", fps=self.framerate)
                    .filter("scale", width, height)
                )
                if self.centercrop:
                    x = int((width - self.size) / 2.0)
                    y = int((height - self.size) / 2.0)
                    cmd = cmd.crop(x, y, self.size, self.size)
                out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
                    capture_stdout=True, quiet=True
                )
            except:
                print("ERROR: ffmpeg error at: {}".format(video_id))
                video = torch.zeros(1)
                return video

            if self.centercrop and isinstance(self.size, int):
                height, width = self.size, self.size
            video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            video = torch.from_numpy(video.astype("float32"))  # video_shape: num_frames x 224 x 224 x3
            video = video.permute(0, 3, 1, 2)  # video_shape: num_frames x 3 x 224 x 224
        else:
            video = torch.zeros(1)
            if not os.path.isfile(video_path):
                print("ERROR: video path not exits", video_path)
            else:
                print("ERROR: UNEXPECTED ERROR", video_path)
        return video

    def __getitem__(self, idx):
        if self.dataset_name == WEBVID:
            video_id = self.csv["videoid"].values[idx]
            video_name = self.csv["path"].values[idx]
        elif self.dataset_name == TGIF:
            video_id = self.csv["video_id"].values[idx]
            video_name = video_id + '.gif'
        elif self.dataset_name == MSVD:
            video_id = self.csv["video_id"].values[idx]
            video_name = self.csv["video_name"].values[idx] + '.avi'
        elif self.dataset_name == ACTIVITYNET:
            video_id = self.csv["video_id"].values[idx]
            video_name = 'v_' + video_id + '.mp4'
        elif self.dataset_name == MSRVTT:
            video_id = 'video' + str(self.csv["video_id"].values[idx])
            video_name = video_id + '.mp4'
        else:
            raise NotImplementedError
        video_path = os.path.join(self.video_main_path, str(video_name))
        output_file = os.path.join(self.features_main_path, str(video_id) + '.npy')

        if self.dataset_name == WEBVID:
            if not (os.path.isfile(output_file)):
                video = self._get_video_clip(video_id, video_path)
            else:
                video = torch.zeros(1)
        else:
            video = self._get_video_clip(video_id, video_path)

        return {"video": video, "input": video_path, "output": output_file}
