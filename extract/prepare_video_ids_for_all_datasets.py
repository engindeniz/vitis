import pandas as pd

MSVD = "MSVD"
TGIF = "TGIF"
ACTIVITY_NET = "ActivityNet"
MSRVTT = "MSRVTT"

DATASETS = {
    MSRVTT: {
        "csv_path_train": "data/MSRVTT-QA/public/train.csv",
        "csv_path_val": "data/MSRVTT-QA/public/val.csv",
        "csv_path_test": "data/MSRVTT-QA/public/test.csv",
        "csv_out": "data/MSRVTT-QA/video_id_list.csv"
    },
    TGIF: {
        "csv_path_train": "data/TGIF-QA/public/train.csv",
        "csv_path_test": "data/TGIF-QA/public/test.csv",
        "csv_out": "data/TGIF-QA/video_id_list.csv"
    },
    MSVD: {
        "csv_path_train": "data/MSVD-QA/youtube_mapping.txt",
        "csv_out": "data/MSVD-QA/video_id_list.csv"
    },
    ACTIVITY_NET: {
        "csv_path_train": "data/ActivityNet-QA/public/train.csv",
        "csv_path_val": "data/ActivityNet-QA/public/val.csv",
        "csv_path_test": "data/ActivityNet-QA/public/test.csv",
        "csv_out": "data/ActivityNet-QA/video_id_list.csv"
    }
}

if __name__ == '__main__':
    for dataset_name, dataset_info in DATASETS.items():
        csv_path_train = dataset_info.get("csv_path_train")
        csv_train = pd.read_csv(csv_path_train)
        csv_out = dataset_info.get("csv_out")
        if dataset_name in [MSRVTT, ACTIVITY_NET, TGIF]:
            csv_path_test = dataset_info.get("csv_path_test")
            csv_test = pd.read_csv(csv_path_test)
            video_id_list_train = csv_train["video_id"].unique().tolist()
            video_id_list_test = csv_test["video_id"].unique().tolist()
            if dataset_name != TGIF:
                csv_path_val = dataset_info.get("csv_path_val")
                csv_val = pd.read_csv(csv_path_val)
                video_id_list_val = csv_val["video_id"].unique().tolist()
                all_video_list = sorted(list(set(video_id_list_train + video_id_list_test + video_id_list_val)))
            else:
                all_video_list = sorted(list(set(video_id_list_train + video_id_list_test)))
            dict_video_list = {"video_id": all_video_list}
            df = pd.DataFrame(dict_video_list)
        elif dataset_name == MSVD:
            df = pd.read_csv(csv_path_train, sep=" ", header=None, names=["video_name", "video_id"])
            df['video_id'] = df['video_id'].str.replace('vid', '')
        else:
            raise NotImplementedError
        df.to_csv(csv_out, index=False)
