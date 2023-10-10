## Zero-Shot and Few-Shot Video Question Answering with Multi-Modal Prompts

[//]: # ([Deniz Engin]&#40;https://engindeniz.github.io/&#41; and [Yannis Avrithis]&#40;https://avrithis.net/&#41;, Zero-Shot and Few-Shot Video)

[//]: # (Question Answering with Multi-Modal Prompts, ICCV 2023 CLVL Workshop.)

[Project page](https://engindeniz.github.io/vitis) | [arXiv](https://arxiv.org/abs/2309.15915)

[//]: # (---)

[//]: # (### Model Overview)

![Model](images/model.png?raw=true)

**ViTiS** consists of a _frozen video encoder_, a _visual mapping network_, a _frozen text embedding layer_, a _frozen
language_
_model_ and a _frozen classifier head_. Given input video frames and text, video encoder extracts frame features and the
visual mapping network maps them to the same space as the text embeddings obtained by text embedding layer. Then, the
language model takes the video and text embeddings as input and predicts the masked input tokens.

The **language model** incorporates **learnable text prompts in** the key and value of multi-head-attention and adapter
layers after each
self-attention and feed-forward layer, before LayerNorm.

Our **visual mapping network** consists of a number of layers,
each performing cross-attention between **learnable visual prompts** and **video frame features** followed by
self-attention.

### Setup

To set up a conda environment:

````
conda env create -f vitis.yml 
conda activate vitis
pip install git+https://github.com/openai/CLIP.git
conda update ffmpeg
````

### Data Preparation

This repository contains both ready-to-use data and guidelines for processing raw data.

##### Processed Data

* Download processed downstream datasets
  from [this link](https://drive.google.com/drive/folders/1WuNmK3LsLBGENxgXyIfk8gF37r2NPvIq?usp=sharing) and place them
  in the [data folder](data).
    * Note that datasets are prepared by following [here](https://github.com/antoyang/FrozenBiLM), features are
      extracted by each dataset.
    * Note that subtitles, vocabulary files, and data splits are obtained
      from [here](https://github.com/antoyang/FrozenBiLM).
    * Due to storage limitations, WebVid2M features are unavailable.

##### Raw Data Processing Guidelines

<details>
  <summary>Click for more details.</summary>

* Download the [WebVid2M](https://maxbain.com/webvid-dataset/) and extract it in the `data/WebVid`.
* Download the [MSRVTT-QA & MSVD-QA](https://github.com/xudejing/video-question-answering) and extract it in
  the `data/MSRVTT-QA` and `data/MSVD-QA`.
    * Note that YouTube mapping file should be downloaded
      from [here](https://mega.nz/#!QrowUADZ!oFfW_M5wAFsfuFDEJAIa2BeFVHYO0vxit3CMkHFOSfw) for MSVD-QA dataset.
* Download the [ActivityNet-QA](https://github.com/MILVLG/activitynet-qa) and extract it in the  `data/ActivityNet-QA`.
* Download the [TGIF-FrameQA](https://github.com/YunseokJANG/tgif-qa) and extract it in the `data/TGIF-QA`.
* For all datasets, videos should be placed in the `data/<dataset_name>/videos` folder.
* For all datasets, download subtitles, vocabulary files, and data splits csv files
  from [this link](https://drive.google.com/drive/folders/1WuNmK3LsLBGENxgXyIfk8gF37r2NPvIq?usp=sharing).

##### Feature Extraction for downstream datasets

* Prepare video id list for all datasets:

```
python extract/prepare_video_ids_for_all_datasets.py
```

* Download [CLIP model](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)
  to [checkpoints folder](checkpoints/openai-clip-vit-large-patch14).
* Extract video features for each dataset: `<dataset_name> : {msrvtt | msvd | activitynet | tgif | webvid}`
* Extract video features for each dataset
  paths: `<DATASET_PATH> : {MSRVTT-QA | MSVD-QA | ActivityNet-QA | TGIF-QA | WEBVID}`
* Create `features` folder in the `data/<DATASET_PATH>`

```
python extract/extract_video_features.py --dataset_name <dataset_name> \ 
--feature_extraction_csv data/<DATASET_PATH>/video_id_list.csv \
--feature_extraction_video_main_path data/<DATASET_PATH>/videos \
--feature_extraction_features_main_path data/<DATASET_PATH>/features
```

* Merge video features for each dataset (except webvid):

```
python extract/merge_features.py --dataset <dataset_name> \
--folder data/<DATASET_PATH>/features \ 
--output_path data/<DATASET_PATH>/features/clipvitl14.pth
```

* Merge video features for webvid:

```
python extract/create_hdf5.py
```

</details>

## Pre-training

* Download DebertaV2 model files to `checkpoints/deberta-v2-xlarge` folder
  from [here](https://huggingface.co/microsoft/deberta-v2-xlarge/tree/main).

* To train ViTiS on Webvid2M, run the following code:

```
python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py \
--combine_datasets webvid --combine_datasets_val webvid --save_dir==output_webvid --lr=2e-5 --different_lr_embedding_layers \
--batch_size=16 --batch_size_val=16 --epochs=10  --amp  \ 
--mapping_network_feedforward  --text_prompt_projection_layer
```

The other parameters are set to default. You can also check our paper.
Note that pre-training is done on 8 Tesla V100 GPUs (32 GB).

### Zero-shot evaluation

* Download pre-trained model files to `checkpoints` folder from [here](https://drive.google.com/file/d/1nEMsaAUMWY55yaSbGP73lSpwingAgcf-/view?usp=drive_link).

* To evaluate ViTiS for zero-shot, run the following code:

```
python -m torch.distributed.launch --nproc_per_node 1 --use_env python videoqa.py --combine_datasets <dataset_name> --combine_datasets_val <dataset_name> \
--batch_size_val=32  --amp --mapping_network_feedforward  --text_prompt_projection_layer \
--<dataset_name>_vocab_path=data/<DATASET_PATH>`/vocab1000.json --load checkpoints/vitis_pretraining_zero_shot.pth --eval --test \
 ```

### Few-shot fine-tuning

* Download pre-trained model file to `checkpoints` folder from [here](https://drive.google.com/file/d/1bcHBoqP6smXlrMM1RSJLxd0QeYDwtRuF/view?usp=drive_link).
* Note that zero-shot and few-shot checkpoints are taken from different epoch.
* We choose the vocabulary that yields the best performance on the validation set.
* Note that fine-training is done on 4 Tesla V100 GPUs (32 GB).

#### All trainable model parameters fine-tuned

* For few-shot fine-tuning all trainable params, run the following code:

```
python -m torch.distributed.launch --nproc_per_node 4 --use_env python videoqa.py --combine_datasets <dataset_name> --combine_datasets_val <dataset_name> \
--save_dir==output_few_shot --lr=1e-5 --different_lr_embedding_layers \
--amp --mapping_network_feedforward  --text_prompt_projection_layer \
--batch_size=8 --batch_size_val=32 --epochs=20  --<dataset_name>_vocab_path=data/<DATASET_PATH>`/vocab1000.json   \ 
--load checkpoints/vitis_pretraining_few_shot.pth
```
* Note that the base learning rate is searched over 5 values in the interval [10−5 , 5 × 10−5], while the learning rate for visual and text prompts is kept at 10−3.

#### Only prompts fine-tuned

* Download saved prompts file to `checkpoints` folder from [here](https://drive.google.com/file/d/1UJATR-WDlxJJZ9QRa3rFZrcdD3OG_9wf/view?usp=drive_link).
* For few-shot fine-tuning only prompts, run the following code:

```
python -m torch.distributed.launch --nproc_per_node 4 --use_env python videoqa.py --combine_datasets <dataset_name> --combine_datasets_val <dataset_name> \
--save_dir==output_few_shot --lr=1e-2 --amp --mapping_network_feedforward --batch_size=8 --batch_size_val=32 --epochs=20 \ 
--<dataset_name>_vocab_path=data/<DATASET_PATH>`/vocab1000.json   \ 
--load checkpoints/vitis_pretraining_few_shot.pth --loaded_prompts text --only_finetune_loaded_prompts visual_text \
```
* Note that the base learning rate is searched over 3 values in the interval [10−2 , 3 × 10−2].

### License

This code is released under the Apache License 2.0.

### Acknowledgments

The code is written based on <a href="https://github.com/antoyang/FrozenBiLM" target="_blank">FrozenBiLM</a>. \
The prompt learning code is inspired by <a href="https://github.com/THUDM/P-tuning-v2/" target="_blank">
P-tuning-v2</a>.

### Citation

If this code is helpful for you, please cite the following:

````
@inproceedings{engin_2023_ICCV,
    title={Zero-Shot and Few-Shot Video Question Answering with Multi-Modal Prompts},
    author={Engin, Deniz and Avrithis, Yannis},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    year={2023}
}

````