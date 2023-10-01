import argparse
import datetime
import json
import math
import os
import sys
import time
from collections import namedtuple

import torch
import torch.nn
import torch.optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb
from args import get_args_parser
from datasets.videotext_dataset import videotext_collate_fn_w_tokenizer, build_videotext_dataset
from model.model_utils import build_model, get_tokenizer
from util import dist
from util.metrics import MetricLogger
from util.misc import adjust_learning_rate
from utils import save_json, initialize_seeds


def train_one_epoch(
        model,
        tokenizer,
        data_loader,
        optimizer,
        device,
        epoch,
        args,
        max_norm,
        scaler=None,
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    num_training_steps = int(len(data_loader) * args.epochs)

    if dist.is_main_process():
        p_bar = tqdm(desc="  Training =>", total=len(data_loader))

    # accuracies = []
    for i_batch, batch_dict in enumerate(
            metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        if dist.is_main_process():
            p_bar.update()

        video = batch_dict["video"].to(device)
        video_mask = batch_dict["video_mask"].to(device)
        attention_mask = batch_dict["attention_mask"].to(device)
        inputs = batch_dict["inputs"].to(device)
        labels = batch_dict["labels"].to(device)
        # forward
        with autocast(enabled=args.amp):
            output = model(
                video=video,
                video_mask=video_mask,
                input_ids=inputs,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = output["loss"]

            # reduce losses over all GPUs for logging purposes
            loss_dict = {"mlm_loss": loss}
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

    if dist.is_main_process():
        print("Gradient norms for trainable layers")
        for n, p in model.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                print(n, ": ", p.grad.abs().detach().cpu().mean())

    metric_logger.synchronize_between_processes()

    if dist.is_main_process():
        print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
        model,
        tokenizer,
        data_loader,
        device,
        args,
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Val:"

    for i_batch, batch_dict in enumerate(
            metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        video = batch_dict["video"].to(device)
        video_mask = batch_dict["video_mask"].to(device)
        attention_mask = batch_dict["attention_mask"].to(device)
        inputs = batch_dict["inputs"].to(device)
        labels = batch_dict["labels"].to(device)
        # forward
        with autocast(enabled=args.amp):
            output = model(
                video=video,
                video_mask=video_mask,
                input_ids=inputs,
                attention_mask=attention_mask,
                labels=labels,
            )

            # calculate loss
            loss = output["loss"]
            # reduce losses over all GPUs for logging purposes
            loss_dict = {"mlm_loss": loss}
            loss_dict_reduced = dist.reduce_dict(loss_dict)
            loss_reduced = sum(loss_dict_reduced.values())
            loss_value = loss_reduced.item()

        metric_logger.update(
            loss=loss_value,
            **loss_dict_reduced,
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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
        wandb.init(project=args.project_name, notes=args.wandb_notes,
                   entity="XXX", id=run_name)
        wandb.config.update(args)

    if dist.is_main_process():
        if args.save_dir and not (os.path.isdir(args.save_dir)):
            os.mkdir(os.path.join(args.save_dir))
        print(args)

        cfg_name = os.path.join(args.save_dir, "args.json")
        args_dict = vars(args)
        save_json(args_dict, cfg_name, save_pretty=True)

    device = torch.device(args.device)

    # Fix seeds
    initialize_seeds(args.seed)

    # Build model
    model = build_model(args)
    model.to(device)
    tokenizer = get_tokenizer(args)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if dist.is_main_process():
        if args.wandb != "disabled":
            wandb.config.update({'number of trainable params': n_parameters})

    # Set up optimizer

    if args.different_lr_embedding_layers:
        params_for_optimization = list(p for p in model.parameters() if
                                       p.requires_grad)
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

    # Set up dataloaders
    if not args.eval:
        if "webvid" in args.combine_datasets:
            dataset_train = build_videotext_dataset("train", args)
            sampler_train = (
                DistributedSampler(dataset_train)
                if args.distributed
                else torch.utils.data.RandomSampler(dataset_train)
            )
            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=True
            )
            dataloader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=videotext_collate_fn_w_tokenizer,
                num_workers=args.num_workers,
            )
        else:
            raise NotImplementedError

    nt = namedtuple(
        typename="data",
        field_names=["dataset_name", "dataloader"],
    )

    tuples = []
    if "webvid" in args.combine_datasets_val:
        webvid_dataset_val = build_videotext_dataset("val", args)
        webvid_sampler_val = (
            DistributedSampler(webvid_dataset_val, shuffle=False)
            if args.distributed
            else torch.utils.data.SequentialSampler(webvid_dataset_val)
        )
        webvid_dataloader_val = DataLoader(
            webvid_dataset_val,
            batch_size=args.batch_size_val,
            sampler=webvid_sampler_val,
            collate_fn=videotext_collate_fn_w_tokenizer,
            num_workers=args.num_workers,
        )
        tuples.append(nt(dataset_name="webvid", dataloader=webvid_dataloader_val))
    else:
        raise NotImplementedError

    # Load pretrained checkpoint
    if args.load:
        if dist.is_main_process():
            print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        if args.resume and not args.eval:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.amp:
                scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        val_stats = {}
        for i, item in enumerate(tuples):
            curr_val_stats = evaluate(
                model=model,
                tokenizer=tokenizer,
                data_loader=item.dataloader,
                device=device,
                args=args,
            )
            val_stats.update(
                {item.dataset_name + "_" + k: v for k, v in curr_val_stats.items()}
            )

        log_stats = {
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": args.start_epoch,
            "n_parameters": n_parameters,
        }

        if args.save_dir and dist.is_main_process():
            json.dump(
                log_stats, open(os.path.join(args.save_dir, "log_stats.json"), "w")
            )
        return

    # Run training and evaluates after every --eval_skip epochs
    if dist.is_main_process():
        print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if dist.is_main_process():
            print(f"Starting epoch {epoch}")
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model=model,
            tokenizer=tokenizer,
            data_loader=dataloader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            max_norm=args.clip_max_norm,
            scaler=scaler,
        )
        if args.wandb != "disabled" and dist.is_main_process():
            log_dict = {'Train Loss': train_stats['loss'],
                        'Base Layers LR': train_stats['lr'], }
            if args.different_lr_embedding_layers:
                log_dict['Embedding Layers LR'] = train_stats['embedding_layer_lr']

            wandb.log(log_dict, step=epoch)

        if args.save_dir and (epoch + 1) % args.eval_skip == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint{epoch:04}.pth")
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

        if (epoch + 1) % args.eval_skip == 0:
            val_stats = {}
            for i, item in enumerate(tuples):
                print(f"Evaluating {item.dataset_name}")

                curr_val_stats = evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    data_loader=item.dataloader,
                    device=device,
                    args=args,
                )
                val_stats.update(
                    {item.dataset_name + "_" + k: v for k, v in curr_val_stats.items()}
                )
                if args.wandb != "disabled" and dist.is_main_process():
                    wandb.log({'Val Loss': curr_val_stats["mlm_loss"]}, step=epoch)
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

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


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
