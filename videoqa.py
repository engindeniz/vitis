import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from collections import namedtuple
from functools import reduce
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb
from args import get_args_parser
from constants import PERCEIVER, VISUAL_TEXT, TEXT, VISUAL, TEXT_PROMPTS, \
    VISUAL_PROMPTS, VISUAL_TEXT_PROMPTS, MAPPING_NETWORK, MLP
from datasets.videoqa_dataset import build_videoqa_dataset, videoqa_collate_fn
from model.model_utils import build_model, get_tokenizer
from util import dist
from util.metrics import MetricLogger
from util.misc import get_mask, adjust_learning_rate
from utils import save_json, initialize_seeds


def calculate_delay(args):
    if args.mapping_network == MLP:
        delay = args.num_frames
    elif args.mapping_network == PERCEIVER:
        delay = args.mapping_prompt_num_tokens
    else:
        raise NotImplementedError
    return delay


def calculate_video_size(args, video):
    if args.mapping_network == MLP:
        video_size = video.size(1)
    elif args.mapping_network == PERCEIVER:
        video_size = args.mapping_prompt_num_tokens
    else:
        raise NotImplementedError
    return video_size


def train_one_epoch(
        model: torch.nn.Module,
        tokenizer,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        dataset_name,
        args,
        num_training_steps_per_epoch,
        max_norm: float = 0,
        scaler=None,
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    num_training_steps = int(len(data_loader) * args.epochs)

    if dist.is_main_process():
        p_bar = tqdm(desc="  Training =>", total=len(data_loader))

    for i_batch, batch_dict in enumerate(
            metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        if dist.is_main_process():
            p_bar.update()

        video = batch_dict["video"].to(device)
        video_len = batch_dict["video_len"]
        video_size = calculate_video_size(args, video)

        video_mask = get_mask(video_len, video_size).to(device)
        text = batch_dict["text"]
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=args.max_tokens_train,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        inputs = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # forward
        with autocast(enabled=args.amp):
            output = model(
                video=video,
                video_mask=video_mask,
                input_ids=inputs,
                attention_mask=attention_mask,
            )

            delay = calculate_delay(args)
            logits = output["logits"][:, delay: encoded["input_ids"].size(1) + delay][
                encoded["input_ids"] == tokenizer.mask_token_id
                ]

            answer_id = batch_dict["answer_id"].to(device)

            loss = F.cross_entropy(logits, answer_id)

            loss_dict = {"cls_loss": loss}

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = dist.reduce_dict(loss_dict)
            loss_reduced = sum(loss_dict_reduced.values())
            loss_value = loss_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

        optimizer.zero_grad()

        if args.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        gradient_norm = None
        if max_norm > 0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        if args.amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        adjust_learning_rate(
            optimizer,
            curr_step=epoch * len(data_loader) + i_batch,
            num_training_steps=num_training_steps,
            args=args,
        )

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if args.different_lr_embedding_layers:
            metric_logger.update(embedding_layer_lr=optimizer.param_groups[1]["lr"])

        if gradient_norm:
            metric_logger.update(gradient_norm=gradient_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if dist.is_main_process():
        print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        tokenizer,
        data_loader,
        device: torch.device,
        dataset_name,
        args,
        thresholds=[1, 10],
        split="test",
        type_map={0: "all"},
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"{split}:"

    res = {}
    val_loss_list = []
    for i_batch, batch_dict in enumerate(
            metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        video = batch_dict["video"].to(device)
        video_len = batch_dict["video_len"]
        video_size = calculate_video_size(args, video)

        video_mask = get_mask(video_len, video_size).to(device)
        text = batch_dict["text"]
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=args.max_tokens_val,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if (
                not args.suffix and not args.use_context
        ):  # remove sep token if not using the suffix
            attention_mask[input_ids == tokenizer.sep_token_id] = 0
            input_ids[input_ids == tokenizer.sep_token_id] = tokenizer.pad_token_id

        with autocast(enabled=args.amp):
            output = model(
                video=video,
                video_mask=video_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        logits = output["logits"]
        delay = calculate_delay(args)

        logits = logits[:, delay: encoded["input_ids"].size(1) + delay][
            encoded["input_ids"] == tokenizer.mask_token_id
            ]  # get the prediction on the mask token

        logits = logits.softmax(-1)
        topk_aids = torch.topk(logits, max(thresholds), -1).indices

        answer_id, qids = batch_dict["answer_id"].to(device), batch_dict["qid"]
        types = batch_dict["type"]
        if "sub" in batch_dict:
            subs = batch_dict["sub"]
        else:
            subs = [0] * len(types)

        answer_id_expanded = answer_id.view(-1, 1).expand_as(topk_aids).to(device)

        if len(logits[answer_id > -1]) > 0:
            loss = F.cross_entropy(logits[answer_id > -1], answer_id[answer_id > -1])
            loss_dict = {"val_loss": loss}

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = dist.reduce_dict(loss_dict)
            loss_reduced = sum(loss_dict_reduced.values())

            loss_value = loss_reduced.item()
        else:
            loss_value = 0
        val_loss_list.append(loss_value)

        agreeings = {}
        for x in thresholds:
            agreeings[x] = topk_aids[:, :x] == answer_id_expanded[:, :x]

        for i, (qid, gt, pred, type, sub) in enumerate(
                zip(qids, answer_id, topk_aids, types, subs)
        ):
            res[qid] = {
                "pred": pred.tolist(),
                "gt": gt.tolist() if dataset_name in ["ivqa", "vqa"] else gt.item(),
                "type": int(type),
                "sub": sub,
            }
            for x in thresholds:
                res[qid][f"acc{x}"] = agreeings[x][i].sum().detach().cpu().item()

        dico = {"acc": agreeings[1].sum() / len(qids)}
        dico_reduced = dist.reduce_dict(dico)
        acc_value = dico_reduced["acc"].item()
        metric_logger.update(acc=acc_value)

    all_res = dist.all_gather(res)
    results = reduce(lambda a, b: a.update(b) or a, all_res, {})
    assert len(results) == len(data_loader.dataset)
    out = {}
    for x in thresholds:
        out[f"acc{x}"] = sum(results[qid][f"acc{x}"] for qid in results) / len(results)
    if type_map is not None and len(type_map) > 1:
        acc_type = {
            type_map[i]: 0 if len([x for x in results.values() if x["type"] == i]) == 0 else sum(
                results[qid][f"acc1"] for qid in results if results[qid]["type"] == i
            )
                                                                                             / len(
                [x for x in results.values() if x["type"] == i])
            for i in type_map
        }
    n_sub = len([x for x in results.values() if x["sub"]])
    if n_sub:
        acc_sub = (
                sum(results[qid][f"acc1"] for qid in results if results[qid]["sub"]) / n_sub
        )
    if dist.is_main_process():
        print(dataset_name)
        out['val_loss'] = torch.mean(torch.Tensor(val_loss_list)).item()

        for x in thresholds:
            print(f"{split} acc{x}: {out[f'acc{x}']: .2%}")
        print(f"{split} loss: {out['val_loss']: .4}")
        if type_map is not None and len(type_map) > 1:
            for x in acc_type:
                print(f"acc {x}: {acc_type[x]: .2%}")
            out.update(acc_type)
        if n_sub:
            print(f"acc sub: {acc_sub: .2%}; proportion {n_sub / len(results): .2%}")
            out["acc_sub"] = acc_sub

    return results, out


def seed_worker(worker_id):
    """
    Dataloader seed
    Borrowed from https://pytorch.org/docs/stable/notes/randomness.html """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(args):
    # Init distributed mode
    dist.init_distributed_mode(args)

    if args.wandb != "disabled" and dist.is_main_process():
        os.environ["WANDB_API_KEY"] = "XXXX"
        os.environ["WANDB_MODE"] = args.wandb
        if hasattr(args, "job_id"):
            run_name = args.run_name + "_job_id_" + str(args.job_id)
        else:
            run_name = args.run_name
        wandb.init(project=args.project_name, notes=args.wandb_notes, tags=["videoqa"],
                   entity="XXXX", id=run_name)
        wandb.config.update(args)

    if dist.is_main_process():

        if args.save_dir and not (os.path.isdir(args.save_dir)):
            os.makedirs(os.path.join(args.save_dir), exist_ok=True)
        print(args)

        cfg_name = os.path.join(args.save_dir, "args.json")
        args_dict = vars(args)
        save_json(args_dict, cfg_name, save_pretty=True)

    device = torch.device(args.device)

    # Fix seeds
    initialize_seeds(args.seed)

    tokenizer = get_tokenizer(args)

    g = torch.Generator()
    g.manual_seed(args.seed)

    nt = namedtuple(
        typename="data",
        field_names=[
            "dataset_name",
            "dataloader_test",
            "dataloader_val",
            "dataloader_train",
        ],
    )

    tuples = []
    for dset_name in args.combine_datasets_val:
        dataset_test = build_videoqa_dataset(
            dset_name,
            "val" if (args.eval and not args.test) else "test",
            args,
            tokenizer,
        )
        sampler_test = (
            DistributedSampler(dataset_test, shuffle=False)
            if args.distributed
            else torch.utils.data.SequentialSampler(dataset_test)
        )
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=args.batch_size_val,
            sampler=sampler_test,
            collate_fn=videoqa_collate_fn,
            num_workers=args.num_workers,
        )

        dataset_val = build_videoqa_dataset(dset_name, "val", args, tokenizer)
        sampler_val = (
            DistributedSampler(dataset_val, shuffle=False)
            if args.distributed
            else torch.utils.data.SequentialSampler(dataset_val)
        )
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size_val,
            sampler=sampler_val,
            collate_fn=videoqa_collate_fn,
            num_workers=args.num_workers,
        )

        if not args.eval:
            dataset_train = build_videoqa_dataset(dset_name, "train", args, tokenizer)

            sampler_train = (
                DistributedSampler(dataset_train)
                if args.distributed
                else torch.utils.data.RandomSampler(dataset_train)
            )

            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=False
            )
            dataloader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=videoqa_collate_fn,
                num_workers=args.num_workers,
                worker_init_fn=seed_worker,
                generator=g,
            )
        else:
            dataloader_train = None

        tuples.append(
            nt(
                dataset_name=dset_name,
                dataloader_test=dataloader_test,
                dataloader_val=dataloader_val,
                dataloader_train=dataloader_train,
            )
        )

    args.n_ans = len(dataloader_test.dataset.a2id)
    model = build_model(args)

    if args.load:
        if dist.is_main_process():
            print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        if args.loaded_prompts:
            if args.loaded_prompts == TEXT:
                original_keys_len = len(checkpoint['model'].keys())
                removed_keys = [key for key in checkpoint['model'].keys() if
                                "prefix" in key]
                [checkpoint["model"].pop(i) for i in removed_keys]
                assert original_keys_len == len(checkpoint['model'].keys()) + len(removed_keys)
                model.prefix_encoder.embedding.data = torch.load(
                    os.path.join(args.saved_prompts_path, "saved_text_prompt_embeddings.pth")).to(torch.float32)
            elif args.loaded_prompts == VISUAL_TEXT:
                original_keys_len = len(checkpoint['model'].keys())
                removed_keys = [key for key in checkpoint['model'].keys() if
                                "prefix" in key or "transformer_video.prompt_proj" in key or "prompt_embeddings" in key]
                [checkpoint["model"].pop(i) for i in removed_keys]
                assert original_keys_len == len(checkpoint['model'].keys()) + len(removed_keys)
                model.prefix_encoder.embedding.data = torch.load(
                    os.path.join(args.saved_prompts_path, "saved_text_prompt_embeddings.pth")).to(torch.float32)
                model.deberta.embeddings.transformer_video.mapping_prompt_embeddings.data = torch.load(
                    os.path.join(args.saved_prompts_path, "saved_mapping_network_prompt_embeddings.pth")).to(
                    torch.float32)
                model.deberta.embeddings.transformer_video.mapping_deep_prompt_embeddings.data = torch.load(
                    os.path.join(args.saved_prompts_path,
                                 "saved_mapping_network_deep_prompt_embeddings.pth")).to(torch.float32)
            elif args.loaded_prompts == VISUAL:
                original_keys_len = len(checkpoint['model'].keys())
                removed_keys = [key for key in checkpoint['model'].keys() if
                                "transformer_video.prompt_proj" in key or "prompt_embeddings" in key]
                [checkpoint["model"].pop(i) for i in removed_keys]
                assert original_keys_len == len(checkpoint['model'].keys()) + len(removed_keys)
                model.deberta.embeddings.transformer_video.mapping_prompt_embeddings.data = torch.load(
                    os.path.join(args.saved_prompts_path, "saved_mapping_network_prompt_embeddings.pth")).to(
                    torch.float32)
                model.deberta.embeddings.transformer_video.mapping_deep_prompt_embeddings.data = torch.load(
                    os.path.join(args.saved_prompts_path,
                                 "saved_mapping_network_deep_prompt_embeddings.pth")).to(torch.float32)
            else:
                raise NotImplementedError

        model.load_state_dict(checkpoint["model"], strict=False)

    model.to(device)

    if args.only_finetune_loaded_prompts and args.trained_modules:
        raise NotImplementedError

    if args.trained_modules:
        if args.trained_modules == TEXT_PROMPTS:
            for n, p in model.named_parameters():
                if ("prefix_encoder" in n):
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
        elif args.trained_modules == VISUAL_PROMPTS:
            for n, p in model.named_parameters():
                if not ("prefix_encoder" in n) and ("prompt" in n):
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
        elif args.trained_modules == VISUAL_TEXT_PROMPTS:
            for n, p in model.named_parameters():
                if ("prefix_encoder" in n) or ("prompt" in n):
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
        elif args.trained_modules == MAPPING_NETWORK:
            for n, p in model.named_parameters():
                if ("transformer_video" in n) and not ("prompt" in n):
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)

        else:
            raise NotImplementedError

    if args.only_finetune_loaded_prompts:
        trainable_layers = []
        trainable_layer_names = []
        if args.only_finetune_loaded_prompts == TEXT:
            trainable_layers.append("prefix_encoder.embedding")
        elif args.only_finetune_loaded_prompts == VISUAL:
            trainable_layers.append("prompt_embeddings")
        elif args.only_finetune_loaded_prompts == VISUAL_TEXT:
            trainable_layers.append("prefix_encoder.embedding")
            trainable_layers.append("mapping_prompt_embeddings")
        else:
            raise NotImplementedError

        for i in trainable_layers:
            for n, p in model.named_parameters():
                if i in n:
                    trainable_layer_names.append(n)

        for n, p in model.named_parameters():
            if n in trainable_layer_names:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if dist.is_main_process():
        if not args.eval:
            if args.wandb != "disabled":
                wandb.config.update({'number of trainable params': n_parameters})

    # Set up optimizer
    if args.different_lr_embedding_layers:
        params_for_optimization = list(p for p in model.parameters() if
                                       p.requires_grad)
        # if args.different_lr_embedding_layers_for_text_prompts:
        #     params_from_text_prompt_embedding_layers_for_optimization = [p for n, p in model.named_parameters() if
        #                                                                  p.requires_grad and ("prefix_encoder" in n)]
        #     names_from_text_prompt_embedding_layers_for_optimization = [n for n, p in model.named_parameters() if
        #                                                                 p.requires_grad and ("prefix_encoder" in n)]
        #     params_from_embedding_layers_for_optimization = [p for n, p in model.named_parameters() if
        #                                                      p.requires_grad and not ("prefix_encoder" in n) and (
        #                                                              "prompt" in n)]
        #     names_from_embedding_layers_for_optimization = [n for n, p in model.named_parameters() if
        #                                                     p.requires_grad and not ("prefix_encoder" in n) and (
        #                                                             "prompt" in n)]
        #     params_from_base_model_for_optimization = [p for n, p in model.named_parameters() if
        #                                                p.requires_grad and n not in (
        #                                                        names_from_embedding_layers_for_optimization + names_from_text_prompt_embedding_layers_for_optimization)]
        #     assert len(params_for_optimization) == len(params_from_embedding_layers_for_optimization) + len(
        #         params_from_base_model_for_optimization) + len(
        #         params_from_text_prompt_embedding_layers_for_optimization)
        # else:
        params_from_embedding_layers_for_optimization = [p for n, p in model.named_parameters() if
                                                         p.requires_grad and (
                                                                 ("prefix_encoder" in n) or ("prompt" in n))]
        names_from_embedding_layers_for_optimization = [n for n, p in model.named_parameters() if
                                                        p.requires_grad and (
                                                                ("prefix_encoder" in n) or ("prompt" in n))]
        params_from_base_model_for_optimization = [p for n, p in model.named_parameters() if
                                                   p.requires_grad and n not in names_from_embedding_layers_for_optimization]
        assert len(params_for_optimization) == len(params_from_embedding_layers_for_optimization) + len(
            params_from_base_model_for_optimization)
    else:
        params_for_optimization = list(p for p in model.parameters() if
                                       p.requires_grad)
    if args.optimizer == 'adam':
        if args.different_lr_embedding_layers:
            # if args.different_lr_embedding_layers_for_text_prompts:
            #     optimizer = torch.optim.Adam([
            #         {'params': params_from_base_model_for_optimization},
            #         {'params': params_from_embedding_layers_for_optimization, 'lr': args.embedding_layer_lr},
            #         {'params': params_from_text_prompt_embedding_layers_for_optimization,
            #          'lr': args.text_prompt_embedding_layer_lr},
            #     ],
            #         lr=args.lr,
            #         betas=(args.beta1, args.beta2),
            #         weight_decay=args.weight_decay,
            #     )
            # else:
            optimizer = torch.optim.Adam([
                {'params': params_from_base_model_for_optimization},
                {'params': params_from_embedding_layers_for_optimization, 'lr': args.embedding_layer_lr}],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                params_for_optimization,
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
            )

    else:
        raise NotImplementedError

    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    # Load pretrained checkpoint
    if args.load:
        if args.resume and not args.eval:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.amp:
                scaler.load_state_dict(checkpoint['scaler'])

    for i, item in enumerate(tuples):
        aid2tokid = torch.zeros(
            len(item.dataloader_test.dataset.a2id), args.max_atokens
        ).long()
        for a, aid in item.dataloader_test.dataset.a2id.items():
            tok = torch.tensor(
                tokenizer(
                    a,
                    add_special_tokens=False,
                    max_length=args.max_atokens,
                    truncation=True,
                    padding="max_length",
                )["input_ids"],
                dtype=torch.long,
            )
            aid2tokid[aid] = tok

        model_device = next(model.parameters()).device
        model.set_answer_embeddings(
            aid2tokid.to(model_device), freeze_last=args.freeze_last
        )  # init answer embedding module

        if not args.eval:
            if dist.is_main_process():
                print("Start training")

            start_time = time.time()
            best_epoch = args.start_epoch
            best_acc = 0

            for epoch in range(args.start_epoch, args.epochs):

                if dist.is_main_process():
                    print(f"Starting epoch {epoch}")
                if args.distributed:
                    sampler_train.set_epoch(epoch)
                    total_batch_size = args.batch_size * args.world_size
                else:
                    total_batch_size = args.batch_size
                num_training_steps_per_epoch = len(dataset_train) // total_batch_size
                train_stats = train_one_epoch(
                    model=model,
                    tokenizer=tokenizer,
                    data_loader=item.dataloader_train,
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    dataset_name=item.dataset_name,
                    args=args,
                    num_training_steps_per_epoch=num_training_steps_per_epoch,
                    max_norm=args.clip_max_norm,
                    scaler=scaler,
                )
                if args.wandb != "disabled" and dist.is_main_process():
                    log_dict = {'Train Loss': train_stats['loss'],
                                'Train cls loss': train_stats['cls_loss'],
                                'Base Layers LR': train_stats['lr']}
                    if args.different_lr_embedding_layers:
                        log_dict['Embedding Layers LR'] = train_stats['embedding_layer_lr']
                    # if args.different_lr_embedding_layers_for_text_prompts:
                    #     log_dict['Text Prompt Embedding Layers LR'] = train_stats['text_prompt_embedding_layer_lr']

                    wandb.log(log_dict, step=epoch)

                if (epoch + 1) % args.eval_skip == 0:
                    val_stats = {}
                    for i, item in enumerate(tuples):
                        print(f"Validating {item.dataset_name}")

                        curr_val_stats, out = evaluate(
                            model=model,
                            tokenizer=tokenizer,
                            data_loader=item.dataloader_val,
                            device=device,
                            dataset_name=item.dataset_name,
                            args=args,
                            split="val",
                            type_map=item.dataloader_val.dataset.type_map,
                        )
                        val_stats.update(
                            {item.dataset_name + "_" + k: v for k, v in out.items()}
                        )

                        if args.wandb != "disabled" and dist.is_main_process():
                            wandb.log({'Val Acc1': out["acc1"]}, step=epoch)
                            wandb.log({'Val Loss': out["val_loss"]}, step=epoch)

                        if out["acc1"] > best_acc:
                            best_epoch = epoch
                            best_acc = out["acc1"]

                            if dist.is_main_process() and args.save_dir:
                                checkpoint_path = os.path.join(
                                    args.save_dir, f"best_model.pth"
                                )
                                if args.amp:
                                    scaler_state_dict = scaler.state_dict()
                                else:
                                    scaler_state_dict = None
                                dist.save_on_master(
                                    {
                                        "model": model.state_dict(),
                                        "optimizer": optimizer.state_dict(),
                                        "epoch": epoch,
                                        "args": args,
                                        "scaler": scaler_state_dict,
                                    },
                                    checkpoint_path,
                                )
                                json.dump(
                                    curr_val_stats,
                                    open(
                                        os.path.join(
                                            args.save_dir,
                                            item.dataset_name + "_val.json",
                                        ),
                                        "w",
                                    ),
                                )
                                json.dump(
                                    {"acc": best_acc, "ep": epoch},
                                    open(
                                        os.path.join(
                                            args.save_dir,
                                            item.dataset_name + "_acc_val.json",
                                        ),
                                        "w",
                                    ),
                                )
                        if args.wandb != "disabled" and dist.is_main_process():
                            wandb.log({'Best Val Acc1': best_acc, 'Best Val Epoch': best_epoch}, step=epoch)
                else:
                    val_stats = {}

                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"val_{k}": v for k, v in val_stats.items()},
                    "epoch": epoch,
                    "n_parameters": n_parameters,
                }

                if args.save_dir and dist.is_main_process():
                    with open(os.path.join(args.save_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                    checkpoint_path = os.path.join(args.save_dir, f"ckpt.pth")
                    dist.save_on_master(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        },
                        checkpoint_path,
                    )

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Training time {}".format(total_time_str))
            # load best ckpt
            if dist.is_main_process() and args.save_dir:
                print(f"loading best checkpoint from epoch {best_epoch}")
            if args.save_dir:
                if args.distributed:
                    torch.distributed.barrier()  # wait all processes
                checkpoint = torch.load(
                    os.path.join(args.save_dir, f"best_model.pth"),
                    map_location="cpu",
                )
                model.load_state_dict(checkpoint["model"], strict=False)

        results, out = evaluate(
            model=model,
            tokenizer=tokenizer,
            data_loader=item.dataloader_test,
            device=device,
            dataset_name=item.dataset_name,
            args=args,
            split="val" if (args.eval and not args.test) else "test",
            type_map=item.dataloader_test.dataset.type_map,
        )

        if args.save_dir and dist.is_main_process():
            json.dump(
                results,
                open(os.path.join(args.save_dir, item.dataset_name + ".json"), "w"),
            )
            json.dump(
                out,
                open(
                    os.path.join(args.save_dir, item.dataset_name + "test_set_summary.json"), "w"
                ),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()

    if args.save_dir:
        if "SLURM_JOBID" in os.environ:
            args.job_id = os.environ["SLURM_JOBID"]
            args.save_dir = os.path.join(args.presave_dir, args.save_dir + "_job_id_" + args.job_id)
        else:
            args.save_dir = os.path.join(args.presave_dir, args.save_dir)
    args.text_encoder_model_path = os.path.join("checkpoints", args.text_encoder)

    main(args)
