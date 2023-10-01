import json
import os

import h5py
import numpy as np
import pandas as pd
import tqdm


class FastH5:
    def __init__(self, h5file, total_data):
        self.file = h5file
        self.group_size = int(total_data ** 0.5)
        self.counter = 0
        self.group_number = 0
        self.current_group = h5file.create_group(f"group_{self.group_number}")
        self.group_instance_dict = {}

    def create_dataset(self, name, data, **kwargs):
        if self.counter > self.group_size:
            self.counter = 0
            self.group_number += 1
            self.current_group = self.file.create_group(f"group_{self.group_number}")

        self.counter += 1
        self.current_group.create_dataset(name, data=data, **kwargs)
        self.group_instance_dict[name] = f'group_{self.group_number}'
        return None

    def close(self):
        self.file.close()
        return None


def save_h5py_per_group(main_features_path, video_id_list, base_dir):
    dataset_size = len(video_id_list)
    print("Dataset size: ", dataset_size)

    hf = h5py.File(
        f'{base_dir}webvid_extracted_features.hdf5', 'a')  # open a hdf5 file
    fast5 = FastH5(hf, int(dataset_size))

    for i in tqdm.tqdm(range(int(dataset_size))):
        video_id = str(video_id_list[i])
        video_feature_path = os.path.join(main_features_path, str(video_id) + '.npy')
        if os.path.isfile(video_feature_path):
            video_feature = np.load(video_feature_path)
            fast5.create_dataset(video_id, video_feature)
    hf.close()  # close the hdf5 file
    with open(f"{base_dir}group_image_ids.json", "w") as f:
        json.dump(fast5.group_instance_dict, f)


if __name__ == '__main__':
    # train
    main_features_path = "data/WebVid/train/features/"
    csv_file = "data/WebVid/train/results_2M_train.csv"
    base_dir = "data/WebVid/train/"

    # val
    # main_features_path = "data/WebVid/val/features/"
    # csv_file = "data/WebVid/val/results_2M_val.csv"
    # base_dir = "data/WebVid/val/"

    csv = pd.read_csv(csv_file)
    csv.dropna(inplace=True)
    video_id_list = csv["videoid"].values
    video_id_list = list(set(video_id_list))
    save_h5py_per_group(main_features_path, video_id_list, base_dir)
