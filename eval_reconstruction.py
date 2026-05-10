import math
import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision import transforms
import wandb
from datetime import datetime

from dataset_reconstruction import TargetDomainDataset
from model import get_model
from utils import (
    get_multiclass_metrics,
    add_dict_to_argparser,
    torch_batch_timesteps_multinomial,
)
import guided_diffusion.guided_diffusion.gaussian_diffusion as gd
import torchvision.transforms.functional as Fv

# from torchvision.models import swin_t, Swin_T_Weights
from mmcls.models import build_classifier
from mmcv.runner import load_checkpoint


def eval():

    print("\n[START EXPERIMENT]\n")
    SEED = 9
    torch.manual_seed(SEED)
    import random

    random.seed(SEED)
    import numpy as np

    np.random.seed(SEED)

    # dealing with args
    defaults = {
        "batch_size": 64,
        "dataset_source_dir": "./imagenet-val",
        "dataset_target_dir": "./imagenet_c/digital",
        "dataset_target_domains": "jpeg_compression",
        "dataset_target_domain_severity": 5,
        "load_checkpoint": "./checkpoints/discriminator.pt",
        "checkpoint_path": "./checkpoints",
        "classifier_weights": "./checkpoints/swin_tiny_224_b16x64_300e_imagenet.pth",
        "project_name": "eval_reconstruction",
        "eval_split": "val",  # either 'val' or 'test'
        "noise_step": None,  # apply the same noise level to the input batch, if number
        "eval_type": "custom",
        "eval_threshold": 40,  # used only if eval_type is 'threshold'
        "eval_guidance": False,
        "disc_threshold": 0.5,  # discriminator threshold to use in eval_type 'custom'
        "disc_strategy": "noise",  # what samples the discriminator use to identify the threshold
    }

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()

    assert args.eval_split == "val" or args.eval_split == "test"

    # experiment setup
    os.makedirs(
        args.checkpoint_path, exist_ok=True
    ) 

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    args_dict = vars(args)

    # DATA
    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    args_dict["dataset_target_domains"] = args_dict["dataset_target_domains"].split(
        ","
    )  # from string to list
    dataset_args = {
        k.replace("dataset_", ""): v
        for k, v in args_dict.items()
        if k.startswith("dataset")
    }  # keeps only data args

    val_dataset = TargetDomainDataset(
        split=args.eval_split, transform=transformations, **dataset_args
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    # MODEL

    # discriminator
    if (
        args.eval_type == "custom"
    ):  # init discriminator only when using adaptive scheduling
        discriminator = get_model("unet_enc_atn")

        if args.load_checkpoint is not None:
            assert os.path.isfile(args.load_checkpoint)

            discriminator.load_state_dict(
                torch.load(args.load_checkpoint, weights_only=True)
            )
            print(f"Loading successfully!")

        discriminator.to(DEVICE)

    # diffusion model and unet
    import guided_diffusion.guided_diffusion.gaussian_diffusion as gd
    from guided_diffusion.guided_diffusion.script_util import create_model_and_diffusion
    from guided_diffusion.guided_diffusion import dist_util

    params = {
        "image_size": 256,
        "num_channels": 256,
        "num_res_blocks": 2,
        "num_heads": 4,
        "num_heads_upsample": -1,
        "num_head_channels": 64,
        "attention_resolutions": "32,16,8",
        "channel_mult": "",
        "dropout": 0.0,
        "class_cond": False,
        "use_checkpoint": False,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_fp16": False,
        "use_new_attention_order": False,
        "learn_sigma": True,
        "diffusion_steps": 1000,
        "noise_schedule": "linear",
        "timestep_respacing": "100",
        "use_kl": False,
        "predict_xstart": False,
        "rescale_timesteps": False,
        "rescale_learned_sigmas": False,
    }

    unet, dm = create_model_and_diffusion(**params)

    path_unet_weights = "pretrained_weights/256x256_diffusion_uncond.pt"
    unet.load_state_dict(
        dist_util.load_state_dict(path_unet_weights, map_location="cpu")
    )
    unet.eval()
    unet = unet.type(torch.float32)
    unet = unet.to(DEVICE)

    # classifier
    cfg = dict(
        type="ImageClassifier",
        backbone=dict(
            type="SwinTransformer", arch="tiny", img_size=224, drop_path_rate=0.2
        ),
        neck=dict(type="GlobalAveragePooling"),
        head=dict(
            type="LinearClsHead",
            num_classes=1000,
            in_channels=768,
            init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
            loss=dict(type="LabelSmoothLoss", label_smooth_val=0.1, mode="original"),
            cal_acc=False,
        ),
        init_cfg=[
            dict(type="TruncNormal", layer="Linear", std=0.02, bias=0.0),
            dict(type="Constant", layer="LayerNorm", val=1.0, bias=0.0),
        ],
        train_cfg=dict(
            augments=[
                dict(type="BatchMixup", alpha=0.8, num_classes=1000, prob=0.5),
                dict(type="BatchCutMix", alpha=1.0, num_classes=1000, prob=0.5),
            ]
        ),
    )

    model = build_classifier(cfg)
    checkpoint = load_checkpoint(model, args.classifier_weights, map_location=DEVICE)
    model = model.to(DEVICE)

    # TRAINING
    max_batches = len(val_loader)
    log_frequency = math.floor(max_batches / 100)
    h_loss = 0
    compute_acc, compute_precision, compute_recall, compute_f1 = get_multiclass_metrics(
        "multiclass", 1000, True, DEVICE
    )
    (
        noise_compute_acc,
        noise_compute_precision,
        noise_compute_recall,
        noise_compute_f1,
    ) = get_multiclass_metrics("multiclass", 1000, True, DEVICE)
    h_acc = 0
    h_precision = 0
    h_recall = 0
    h_f1 = 0

    h_noise_acc = 0
    h_noise_precision = 0
    h_noise_recall = 0
    h_noise_f1 = 0

    h_thresholds = True

    print(f"STARTING EVALUATION")
    model.eval()

    for batch, (samples, labels) in enumerate(val_loader):

        # move data to GPU
        labels = labels.type(torch.float32).to(DEVICE)
        samples = (
            samples.type(torch.float32).requires_grad_(args.eval_guidance).to(DEVICE)
        )

        # REFERENCE CLASSIFICATION of currupted image
        pre_samples = Fv.normalize(
            samples, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        out = model(pre_samples, return_loss=False)

        # np out to tensor
        out = torch.stack([torch.tensor(array, device=DEVICE) for array in out])

        # noise stats
        noise_compute_acc.update(out, labels)
        noise_compute_precision.update(out, labels)
        noise_compute_recall.update(out, labels)
        noise_compute_f1.update(out, labels)

        # RECONSTRUCTION

        if args.eval_type == "custom":  # custom threshold

            # threshold range
            max_steps = int(params["timestep_respacing"])
            step = max_steps // 20
            timesteps = torch.arange(0, max_steps, step, device=DEVICE)

            # noise

            disc_samples = samples
            disc_samples = Fv.normalize(
                disc_samples, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )

            stat_imgs = [
                dm.q_sample(disc_samples, timesteps[i]) for i in range(len(timesteps))
            ]
            stat_imgs = torch.concatenate(stat_imgs, dim=0)  # [S * B, C, H, W]

            # compute thresholds
            batch_timesteps = torch.stack(
                [timesteps for _ in range(args.batch_size)]
            )  # [B, S]
            batch_timesteps = batch_timesteps.T.reshape(-1)  # [S, B], [S * B]

            if (
                args.disc_strategy == "pred_xstart"
            ):  # predict corruption from pred_xstart instead of the noise
                with torch.no_grad():
                    out = dm.p_sample(
                        unet,
                        stat_imgs,
                        batch_timesteps,
                        clip_denoised=None,
                        denoised_fn=None,
                        cond_fn=None,
                        model_kwargs=None,
                    )

                    stat_imgs = out["pred_xstart"]

            outs = discriminator(
                stat_imgs, batch_timesteps
            )  # [S * B, C, H, W] -> [S * B, 1]
            outs = outs.reshape([timesteps.shape[0], args.batch_size])  # [S, B]
            outs = torch.moveaxis(outs, 0, 1)  # [B, S]

            if args.disc_strategy == "noise":
                th = torch.where(outs <= float(args.disc_threshold), 0, 1)
                th_idx = torch.min(th, dim=1).indices
            elif args.disc_strategy == "pred_xstart":
                th_idx = torch.argmin(outs[:, : len(timesteps) // 2 + 1], dim=1)
            else:
                raise Exception(
                    "The variable 'disc_startegy' DOES NOT belong to this set of values: [noise, pred_xstart]"
                )
            thresholds = timesteps[th_idx]

            # cleanup
            outs = outs.cpu()
            stat_imgs = stat_imgs.cpu()
            batch_timesteps = batch_timesteps.cpu()
            del outs
            del stat_imgs
            del batch_timesteps

            # cap noise
            CAP = 60
            print(f"before {thresholds=}")
            thresholds_too_big = (
                torch.where(thresholds > CAP, 0, 1) * 0
            )  # for noise greater than CAP
            thresholds_right = torch.where(thresholds < CAP, 1, 0) * thresholds
            thresholds = thresholds_too_big + thresholds_right

            # thresholds = torch.where(thresholds > 101, 1, 0) * 0
            noise_threshold = torch.max(thresholds).item()

            # keep the thresholds for stats
            h_thresholds = (
                thresholds.type(torch.uint16)
                if batch == 0
                else torch.concatenate([h_thresholds, thresholds.type(torch.uint16)])
            )

        elif args.eval_type == "threshold":  # prior threshold
            noise_threshold = args.eval_threshold
            thresholds = torch.tensor(
                [args.eval_threshold for _ in range(args.batch_size)], device=DEVICE
            )
        else:
            raise Exception(f"Unknown eval type '{args.eval_type}'")

        # apply forward
        samples -= 0.5
        samples /= 0.5

        pre_samples = samples.clone()
        samples = dm.q_sample(samples, thresholds)

        print(f"{thresholds=}, {noise_threshold=}")

        # RECONSTRUCTION SAMPLING LOOP

        # guidance
        from resize_right import resize

        D = 4
        scale = 6
        noise = pre_samples
        img = None
        shape = samples.shape
        model_kwargs = {
            "ref_img": pre_samples,
        }

        # generation loop
        for index in range(noise_threshold, 0, -1):
            with torch.set_grad_enabled(args.eval_guidance):
                out = dm.p_sample(
                    unet,
                    samples,
                    thresholds,
                    clip_denoised=None,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=None,
                )

                # guidance
                if args.eval_guidance:
                    shape_u = (shape[0], 3, shape[2], shape[3])
                    shape_d = (shape[0], 3, int(shape[2] / D), int(shape[3] / D))

                    difference = resize(
                        resize(
                            model_kwargs["ref_img"],
                            scale_factors=1.0 / D,
                            out_shape=shape_d,
                        ),
                        scale_factors=D,
                        out_shape=shape_u,
                    ) - resize(
                        resize(
                            out["pred_xstart"], scale_factors=1.0 / D, out_shape=shape_d
                        ),
                        scale_factors=D,
                        out_shape=shape_u,
                    )
                    norm = torch.linalg.norm(difference)
                    norm_grad = torch.autograd.grad(outputs=norm, inputs=samples)[0]
                    out["sample"] -= norm_grad * scale

                # replace only the images with positive thresold
                if args.eval_type == "custom":
                    idxs = torch.where(thresholds > 0)[0]
                    samples[idxs] = out["sample"][idxs]

                    # decrement thresholds and clip to 0
                    thresholds = thresholds - 1
                    thresholds = torch.clamp(thresholds, 0)
                else:
                    samples = out["sample"]
                    thresholds = thresholds - 1
                    thresholds = torch.clamp(thresholds, 0)

        # post classification
        samples /= 2
        samples += 0.5
        samples = Fv.normalize(
            samples, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        out = model(samples, return_loss=False)

        # np out to tensor
        out = torch.stack([torch.tensor(array, device=DEVICE) for array in out])

        # stats
        compute_acc.update(out, labels)
        compute_precision.update(out, labels)
        compute_recall.update(out, labels)
        compute_f1.update(out, labels)

        h_acc = compute_acc.compute().item()
        h_noise_acc = noise_compute_acc.compute().item()
        print(
            f"[{batch}/{max_batches} - { batch/max_batches * 100: .1f}%] acc: {h_acc: .6f} noise_acc: {h_noise_acc: .6f}"
        )

    h_acc = compute_acc.compute().item()
    h_precision = compute_precision.compute().item()
    h_recall = compute_recall.compute().item()
    h_f1 = compute_f1.compute().item()

    h_noise_acc = noise_compute_acc.compute().item()
    h_noise_precision = noise_compute_precision.compute().item()
    h_noise_recall = noise_compute_recall.compute().item()
    h_noise_f1 = noise_compute_f1.compute().item()

    metrics = {
        "loss": h_loss,
        "acc": h_acc,
        "precision": h_precision,
        "recall": h_recall,
        "f1": h_f1,
        "noise_acc": h_noise_acc,
        "noise_precision": h_noise_precision,
        "noise_recall": h_noise_recall,
        "noise_f1": h_noise_f1,
    }

    if args.noise_step is not None:
        print(f"noise level: {args.noise_step}")

    final_metrics = f"loss: {metrics['loss']: .6f}, \
            acc: {metrics['acc']: .6f}, \
            f1: {metrics['f1']: .6f}, \
            noise_acc: {metrics['noise_acc']: .6f}, \
            noise_f1: {metrics['noise_f1']: .6f}"

    print(final_metrics)

    # saving the final metrics
    with open(f"{args.checkpoint_path}/metrics.txt", "w+") as file:
        file.write(final_metrics)

    # save all the thresholds
    if args.eval_type == "custom":
        now = datetime.now()
        ckpt_filename = f"{args.checkpoint_path}/ckpt_th_"
        ckpt_filename += f"{now.strftime('%Y%m%d')}_"
        ckpt_filename += f"{str(time.time()).split('.')[0]}"
        np.savez(ckpt_filename, h_thresholds.detach().cpu().numpy())

        print(f"\n Thresholds and final metrics saved at: {ckpt_filename}")


if __name__ == "__main__":
    eval()
