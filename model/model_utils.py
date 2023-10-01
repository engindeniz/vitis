from .multimodal_prompt_deberta_clip import PromptTuningVideoQA
from transformers import DebertaV2Config, DebertaV2Tokenizer


def build_model(args):
    text_config = DebertaV2Config.from_pretrained(
        pretrained_model_name_or_path=args.text_encoder_model_path, local_files_only=True
    )

    model = PromptTuningVideoQA(
        text_config=text_config,
        args=args,
        num_frames=args.num_frames,
        features_dim=args.features_dim,
        visual_encoder=args.visual_encoder,
        freeze_lm=args.freeze_lm,
        freeze_mlm=args.freeze_mlm,
        ds_factor_attn=args.ds_factor_attn,
        ds_factor_ff=args.ds_factor_ff,
        ft_ln=args.ft_ln,
        dropout=args.dropout,
        n_ans=args.n_ans,
        freeze_last=args.freeze_last,
        prompt_type=args.prompt_type,
    )
    return model


def get_tokenizer(args):
    if "deberta" in args.text_encoder:
        tokenizer = DebertaV2Tokenizer.from_pretrained(
            args.text_encoder_model_path, local_files_only=True
        )
    else:
        raise NotImplementedError
    return tokenizer
