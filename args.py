import argparse
import os

PRESAVE_DIR = ""
MODEL_DIR = "checkpoints/"
DATA_DIR = "data/"
SSD_DIR = ""
name2folder = {
    "webvid": "WebVid",
    "msrvtt": "MSRVTT-QA",
    "msvd": "MSVD-QA",
    "activitynet": "ActivityNet-QA",
    "tgif": "TGIF-QA",
}


def get_args_parser():
    parser = argparse.ArgumentParser("Set ViTiS", add_help=False)

    ############################
    # FEATURE EXTRACTION PARAMS #
    ############################
    parser.add_argument(
        "--video_main_path",
        help="video dir",
    )
    parser.add_argument(
        "--feature_extraction_csv",
        type=str,
        help="input csv with columns video_path (input video) and feature_path (output path to feature)",
    )
    parser.add_argument(
        "--feature_extraction_video_main_path", type=str, help="video main path"
    )
    parser.add_argument(
        "--feature_extraction_features_main_path", type=str, help="saved features main path"
    )
    ############################
    # PROMPT PARAMS #
    ############################
    parser.add_argument(
        "--saved_prompts_path",
        default="checkpoints",
        help="Specify saved text prompt path",
    )

    parser.add_argument(
        "--loaded_prompts",
        type=str,
        default=None,
        choices=[None, 'text', 'visual', 'visual_text']
    )

    parser.add_argument(
        "--only_finetune_loaded_prompts",
        type=str,
        default=None,
        choices=[None, 'text', 'visual', 'visual_text']
    )

    ############################
    # Trainable parameters #
    ############################
    parser.add_argument(
        "--trained_modules",
        type=str,
        default=None,
        choices=[None, 'text_prompts', 'visual_prompts', 'visual_text_prompts', 'mapping_network']
    )

    ############################
    # Dataset parameters #
    ############################
    parser.add_argument(
        "--combine_datasets",
        nargs="+",
        help="list of datasets to combine for training",
    )
    parser.add_argument(
        "--combine_datasets_val",
        nargs="+",
        help="list of datasets to combine for eval",

    )

    parser.add_argument(
        "--webvid_train_features_path",
        default="data/WebVid/train/webvid_extracted_features.hdf5")

    parser.add_argument(
        "--webvid_val_features_path",
        default="data/WebVid/val/webvid_extracted_features.hdf5")

    parser.add_argument(
        "--webvid_train_group_image_ids_path",
        default="data/WebVid/train/group_image_ids.json"
    )

    parser.add_argument(
        "--webvid_val_group_image_ids_path",
        default="data/WebVid/val/group_image_ids.json"
    )

    parser.add_argument(
        "--webvid_val_csv_path",
        default="data/WebVid/val/results_2M_val.csv",
    )
    parser.add_argument(
        "--webvid_train_csv_path",
        default="data/WebVid/train/results_2M_train.csv",
    )

    parser.add_argument(
        "--msrvtt_features_path",
        default=os.path.join(DATA_DIR, name2folder["msrvtt"], "clipvitl14.pth"),
    )
    parser.add_argument(
        "--msrvtt_train_csv_path",
        default=os.path.join(DATA_DIR, name2folder["msrvtt"], "train1p.csv"),
    )
    parser.add_argument(
        "--msrvtt_val_csv_path",
        default=os.path.join(DATA_DIR, name2folder["msrvtt"], "val.csv"),
    )
    parser.add_argument(
        "--msrvtt_test_csv_path",
        default=os.path.join(DATA_DIR, name2folder["msrvtt"], "test.csv"),
    )
    parser.add_argument(
        "--msrvtt_vocab_path",
        default=os.path.join(DATA_DIR, name2folder["msrvtt"], "vocab.json"),
    )
    parser.add_argument(
        "--msrvtt_subtitles_path",
        default=os.path.join(DATA_DIR, name2folder["msrvtt"], "subtitles.pkl"),
    )
    parser.add_argument(
        "--msvd_features_path",
        default=os.path.join(DATA_DIR, name2folder["msvd"], "clipvitl14.pth"),
    )
    parser.add_argument(
        "--msvd_train_csv_path",
        default=os.path.join(DATA_DIR, name2folder["msvd"], "train1p.csv"),
    )
    parser.add_argument(
        "--msvd_val_csv_path",
        default=os.path.join(DATA_DIR, name2folder["msvd"], "val.csv"),
    )
    parser.add_argument(
        "--msvd_test_csv_path",
        default=os.path.join(DATA_DIR, name2folder["msvd"], "test.csv"),
    )
    parser.add_argument(
        "--msvd_vocab_path",
        default=os.path.join(DATA_DIR, name2folder["msvd"], "vocab.json"),
    )
    parser.add_argument(
        "--msvd_subtitles_path",
        default=os.path.join(DATA_DIR, name2folder["msvd"], "subtitles.pkl"),
    )
    parser.add_argument(
        "--activitynet_features_path",
        default=os.path.join(DATA_DIR, name2folder["activitynet"], "clipvitl14.pth"),
    )
    parser.add_argument(
        "--activitynet_train_csv_path",
        default=os.path.join(DATA_DIR, name2folder["activitynet"], "train1p.csv"),
    )
    parser.add_argument(
        "--activitynet_val_csv_path",
        default=os.path.join(DATA_DIR, name2folder["activitynet"], "val.csv"),
    )
    parser.add_argument(
        "--activitynet_test_csv_path",
        default=os.path.join(DATA_DIR, name2folder["activitynet"], "test.csv"),
    )
    parser.add_argument(
        "--activitynet_vocab_path",
        default=os.path.join(DATA_DIR, name2folder["activitynet"], "vocab.json"),
    )
    parser.add_argument(
        "--activitynet_subtitles_path",
        default=os.path.join(DATA_DIR, name2folder["activitynet"], "subtitles.pkl"),
    )
    parser.add_argument(
        "--tgif_features_path",
        default=os.path.join(DATA_DIR, name2folder["tgif"], "clipvitl14.pth"),
    )
    parser.add_argument(
        "--tgif_frameqa_train_csv_path",
        default=os.path.join(DATA_DIR, name2folder["tgif"], "train1p.csv"),
    )
    parser.add_argument(
        "--tgif_frameqa_test_csv_path",
        default=os.path.join(DATA_DIR, name2folder["tgif"], "test.csv"),
    )
    parser.add_argument(
        "--tgif_vocab_path",
        default=os.path.join(DATA_DIR, name2folder["tgif"], "vocab.json"),
    )

    # Training hyper-parameters
    parser.add_argument(
        "--mlm_prob",
        type=float,
        default=0.15,
        help="masking probability for the MLM objective",
    )
    parser.add_argument("--lr", default=3e-4, type=float, help="learning rate")

    parser.add_argument(
        "--beta1", default=0.9, type=float, help="Adam optimizer parameter"
    )
    parser.add_argument(
        "--beta2", default=0.95, type=float, help="Adam optimizer parameter"
    )
    parser.add_argument(
        "--batch_size", default=8, type=int, help="batch size used for training"
    )
    parser.add_argument(
        "--batch_size_val",
        default=32,
        type=int,
        help="batch size used for eval",
    )
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument(
        "--epochs", default=20, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--lr_drop",
        default=10,
        type=int,
        help="number of epochs after which the learning rate is reduced when not using linear decay",
    )
    parser.add_argument("--optimizer", default="adam", type=str, choices=['adam', 'sgd'])
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, help="momentum for SGD"
    )
    parser.add_argument(
        "--schedule",
        default="linear_with_warmup",
        choices=["", "linear_with_warmup"],
        help="learning rate decay schedule, default is linear_with_warmup",
    )
    parser.add_argument(
        "--fraction_warmup_steps",
        default=0.1,
        type=float,
        help="fraction of number of steps used for warmup when using linear schedule",
    )

    parser.add_argument(
        "--eval_skip",
        default=1,
        type=int,
        help='do evaluation every "eval_skip" epochs',
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=100,
        help="print log every print_freq iterations",
    )

    # Model parameters
    parser.add_argument(
        "--ft_lm",
        dest="freeze_lm",
        action="store_false",
        help="whether to finetune the weights of the language model",
    )

    parser.add_argument(
        "--model_name",
        default="vitis",
    )

    parser.add_argument(
        "--ds_factor_attn",
        type=int,
        default=8,
        help="downsampling factor for adapter attn",
    )
    parser.add_argument(
        "--ds_factor_ff",
        type=int,
        default=8,
        help="downsampling factor for adapter ff",
    )
    parser.add_argument(
        "--freeze_ln",
        dest="ft_ln",
        action="store_false",
        help="whether or not to freeze layer norm parameters",
    )
    parser.add_argument(
        "--ft_mlm",
        dest="freeze_mlm",
        action="store_false",
        help="whether or not to finetune the mlm head parameters",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="dropout to use in the adapter"
    )
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="whether to train the LM with or without language init",
    )

    parser.add_argument(
        "--n_ans",
        type=int,
        default=0,
        help="number of answers in the answer embedding module, it is automatically set",
    )
    parser.add_argument(
        "--ft_last",
        dest="freeze_last",
        action="store_false",
        help="whether to finetune answer embedding module or not",
    )

    # Run specific
    parser.add_argument(
        "--test",
        action="store_true",
        help="whether to run evaluation on val or test set",
    )
    parser.add_argument(
        "--save_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--presave_dir",
        default=PRESAVE_DIR,
        help="the actual save_dir is an union of presave_dir and save_dir",
    )
    parser.add_argument("--device", default="cuda", help="device to use")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument(
        "--load",
        default="",
        help="path to load checkpoint",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue training if loading checkpoint",
    )
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="only run evaluation")
    parser.add_argument(
        "--num_workers", default=8, type=int, help="number of workers for dataloader"
    )

    # Distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )

    # Video and Text parameters
    parser.add_argument(
        "--max_feats",
        type=int,
        default=10,
        help="maximum number of video features considered, one per frame",
    )
    parser.add_argument(
        "--features_dim",
        type=int,
        default=768,
        help="dimension of the visual embedding space",
    )

    parser.add_argument(
        "--no_context",
        dest="use_context",
        action="store_false",
        help="disables usage of speech",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="maximum number of tokens in the input text prompt",
    )
    parser.add_argument(
        "--max_tokens_train",
        type=int,
        default=128,
        help="maximum number of tokens in the input text prompt",
    )
    parser.add_argument(
        "--max_tokens_val",
        type=int,
        default=128,
        help="maximum number of tokens in the input text prompt",
    )
    parser.add_argument(
        "--max_atokens",
        type=int,
        default=5,
        help="maximum number of tokens in the answer",
    )
    parser.add_argument(
        "--prefix",
        default="",
        type=str,
        help="task induction before question for videoqa",
    )
    parser.add_argument(
        "--suffix",
        default=".",
        type=str,
        help="suffix after the answer mask for videoqa",
    )

    parser.add_argument(
        "--ft_layer_norm",
        action="store_true",
        help="whether to finetune layernorms or not",
    )

    parser.add_argument(
        "--input_res",
        type=int,
        default=224,
        help="image size",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=10,
        help="number of input frames",
    )
    parser.add_argument(
        "--visual_encoder",
        type=str,
        default="clip",
        help="the name of visual encoder",
    )

    parser.add_argument(
        "--deberta_model_path",
        type=str,
        default="./checkpoints/deberta-v2-xlarge/pytorch_model.bin",
    )

    parser.add_argument(
        "--text_encoder",
        type=str,
        default="deberta-v2-xlarge",
        help="the name of text encoder",
    )

    ############################
    # Prompt Parameters #
    ############################
    # general
    parser.add_argument(
        "--prompt_type",
        type=str,
        default='text',
        help="will prompt used, if yes which one",
        choices=[None, 'text']
    )

    # text
    parser.add_argument(
        "--text_prompt_dropout_rate",
        type=float,
        default=0.1,
        help="text prompt dropout rate",
    )

    parser.add_argument(
        "--text_prompt_num_tokens",
        type=int,
        default=10,
        help="number of textual prompt tokens",
    )
    parser.add_argument(
        "--text_prompt_initialization",
        type=str,
        default="kaiming_uniform",
        help="how to initialize prompts",
        choices=["normal", "kaiming_uniform", "kaiming_normal"]
    )
    parser.add_argument(
        "--text_prompt_projection_layer",
        action="store_true",
        help="to have a linear layer to project prompts",
    )
    parser.add_argument(
        "--text_intermediate_proj_layer_dim",
        type=int,
        default=0,
        help="define two layer with given projection dimension",
    )

    parser.add_argument(
        "--text_prompt_projection_factor",
        type=int,
        default=8,
        help="",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="define a dataset for main training",
    )

    parser.add_argument(
        "--emd_context_layer_type",
        type=str,
        default="emd_context_layer_w_text_prompts",
        choices=["emd_context_layer_not_exists", "emd_context_layer_w_text_prompts",
                 "emd_context_layer_wo_text_prompts"]
    )
    ############################
    # Mapping network Parameters#
    ############################
    parser.add_argument(
        "--mapping_network",
        type=str,
        default="perceiver",
        help="video to text mapping network",
        choices=["mlp", "perceiver"]
    )

    parser.add_argument(
        "--mapping_network_num_transformer_layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--mapping_network_num_head_transformer_layers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--mapping_network_transformer_dim_feedforward",
        type=int,
        default=1536,
        help="number of mapping prompt tokens",
    )
    parser.add_argument(
        "--transformer_hidden_size",
        type=int,
        default=1536,
        help="number of mapping prompt tokens",
    )

    parser.add_argument(
        "--mapping_network_feedforward",
        action="store_true",
        help="whether adding prompt to mapping network or not",
    )

    parser.add_argument(
        "--mapping_network_linear_after_transformer",
        action="store_true",
        help="whether adding prompt to mapping network or not",
    )

    parser.add_argument(
        "--mapping_prompt_num_tokens",
        type=int,
        default=10,
        help="number of mapping prompt tokens",
    )

    parser.add_argument(
        "--mapping_prompt_dropout_rate",
        type=float,
        default=0.1,
        help="number of mapping prompt tokens",
    )

    parser.add_argument(
        "--perceiver_self_attn_num_layers",
        type=int,
        default=1,
        help="number of self attention layer in perceiver module if it 0,no self-attention layers",
    )

    ############################
    # Prompt Save #
    ############################
    parser.add_argument(
        "--prompt_save",
        action="store_true",
        help="whether saving prompts or not",
    )
    parser.add_argument(
        "--prompt_save_path",
        type=str,
    )

    ############################
    # WANDB Parameters #
    ############################
    parser.add_argument(
        "--wandb",
        default='disabled',
        help="whether to use wandb or not",
        choices=['disabled', 'online', 'offline'],
    )
    parser.add_argument(
        "--run_name",
        default='run1',
        help="name of the experiment",
    )
    parser.add_argument(
        "--project_name",
        default='my-test-project',
        help="name of the project",
    )
    parser.add_argument(
        "--wandb_notes",
        default='',
        help="note of the project",
    )
    ###########################################################
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Automatic Mixed Precision",
    )

    parser.add_argument(
        "--different_lr_embedding_layers",
        action="store_true",
        help="Whether to set different lr for embedding layer",
    )
    parser.add_argument("--embedding_layer_lr", default=1e-3, type=float, help="learning rate")

    return parser
