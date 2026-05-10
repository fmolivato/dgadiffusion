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

from dataset import SourceTargetDomainDataset
from model import get_model
from utils import get_metrics, add_dict_to_argparser, torch_batch_timesteps_multinomial, random_forward_steps
import guided_diffusion.guided_diffusion.gaussian_diffusion as gd
from guided_diffusion.guided_diffusion.script_util import create_model_and_diffusion

from torch.amp import GradScaler, autocast # half-precision
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# init the diffusion model for the forward steps
dm_params = {
    'image_size': 256, # FIXME originally was 256
    'num_channels': 256,
    'num_res_blocks': 2,
    'num_heads': 4,
    'num_heads_upsample': -1,
    'num_head_channels': 64,
    'attention_resolutions': '32,16,8',
    'channel_mult': '',
    'dropout': 0.0,
    'class_cond': False,
    'use_checkpoint': False,
    'use_scale_shift_norm': True,
    'resblock_updown': True,
    'use_fp16': False, # FIXME originally True
    'use_new_attention_order': False,
    'learn_sigma': True,
    'diffusion_steps': 1000,
    'noise_schedule': 'linear',
    'timestep_respacing': '150',
    'use_kl': False,
    'predict_xstart': False,
    'rescale_timesteps': False,
    'rescale_learned_sigmas': False
}

def main():

    print("\n[START EXPERIMENT]\n")

    # dealing with args
    defaults = {
        "epochs": 5,
        "batch_size": 64,
        "lr": 0.00002,
        "dataset_source_dir": "./imagenet-val",
        "dataset_target_dir": "./imagenet_c/digital",
        "dataset_target_domains": "jpeg_compression,elastic_transform",
        "dataset_target_domain_severity": 5,
        "dataloader_num_worker": 8,
        "keep_checkpoints": True,
        "load_checkpoint": None,
        "checkpoint_path": "./checkpoints",
        "project_name": "adversarial_scheduling",
        "model_name": "unet_enc_atn",
        "model_trainable": False,
        "log_freq_per_epoch": 4,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity', type=str, required=True)
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()

    # experiment setup
    os.makedirs(args.checkpoint_path, exist_ok=True) # makes sure the ckpt folder exists

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    best_val_loss = 10**15

    args_dict = vars(args)
    wandb.init(entity=args.wandb_entity, project=args.project_name, config=args_dict) # track hyperparameter

    _, dm = create_model_and_diffusion(**dm_params)

    # DATA
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
    ])

    args_dict["dataset_target_domains"] = args_dict["dataset_target_domains"].split(",") # from string to list
    dataset_args = {k.replace("dataset_",""):v for k, v in args_dict.items() if k.startswith('dataset')} # keeps only data args

    train_dataset = SourceTargetDomainDataset(split="train", transform=transformations, **dataset_args)
    val_dataset = SourceTargetDomainDataset(split="val", transform=transformations, **dataset_args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.dataloader_num_worker, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.dataloader_num_worker, pin_memory=True)

    # MODEL
    model = get_model(args.model_name, args.model_trainable)

    resumed_steps_offset = 0
    if args.load_checkpoint is not None:
        assert os.path.isfile(args.load_checkpoint)

        model.load_state_dict(torch.load(args.load_checkpoint, weights_only=True))

        # resume steps stat
        resumed_steps_offset = int(args.load_checkpoint.split("_")[-2])
        print(f"Loading successfully! Resumed step {resumed_steps_offset}")

    model.to(DEVICE)

    # TRAINING

    max_batches = len(train_loader)
    log_frequency = math.floor(max_batches / args.log_freq_per_epoch)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    wandb.watch(model) # log="all"

    for epoch in range(args.epochs):

        metrics = {
            "train/loss" : torch.tensor(0.0, device=DEVICE, dtype=torch.float16),
            "val/loss" : 0,
            "val/acc" : 0,
            "val/precision" : 0,
            "val/recall" : 0,
            "val/f1" : 0
        }
        
        model.train()
        for batch, (samples, labels) in enumerate(train_loader):

            #cleaning grads
            opt.zero_grad()

            # move data to GPU
            labels = labels.type(torch.float32).to(DEVICE)
            samples = samples.type(torch.float32).to(DEVICE)

            # data preprocessing
            samples, timesteps = random_forward_steps(dm, int(dm_params["timestep_respacing"]), samples, device=DEVICE)

            samples = samples.type(torch.float32)
            timesteps = timesteps.type(torch.float32)
            
            # inference
            out = model(samples, timesteps)

            # loss
            out = out.reshape(-1)

            epsilon = 0.1
            noise_weights = torch.log((timesteps+epsilon)/1000) * -1
            noise_weights -= noise_weights.min()
            noise_weights /= noise_weights.max()

            # source domain importance is proportinal to the target domain amount
            label_weight = (labels - 1) * (-1 * len(args.dataset_target_domains)) 

            weight = noise_weights * (label_weight + labels)
            loss = torch.nn.functional.binary_cross_entropy(out, labels, weight=weight)

        
            # backward
            loss.backward()
            opt.step()

            # stats
            metrics["train/loss"] += loss.detach()

            if batch % log_frequency == 0:      
                
                if batch != 0: # allows to log the very first train and val numbers
                    metrics["train/loss"] /= log_frequency # running loss normalization
                    metrics["train/loss"] = metrics["train/loss"].item()

                val_out = val(val_loader, model, DEVICE, dm)
                metrics["val/loss"] = val_out[0]
                metrics["val/acc"] = val_out[1]
                metrics["val/precision"] = val_out[2]
                metrics["val/recall"] = val_out[3]
                metrics["val/f1"] = val_out[4]

                print(f"[Epoch {epoch} - {batch}/{max_batches}] \
                    loss: {metrics["train/loss"]: .6f}, \
                    val_loss: {metrics['val/loss']: .6f}, \
                    val_acc: {metrics['val/acc']: .6f}, \
                    val_f1: {metrics['val/f1']: .6f}")
                
                current_step = resumed_steps_offset         # previous run steps
                current_step += epoch*max_batches+batch     # current run steps

                wandb.log(metrics, step=current_step)
                if epoch == 0 and batch == 0: # log wandb run names
                    with open(f"{args.checkpoint_path}/wandb_names.txt", "a+") as file:
                        file.write(f"{wandb.run.name}\n")

                if args.keep_checkpoints and batch != 0:
                    if best_val_loss > metrics["val/loss"]:
                        best_val_loss = metrics["val/loss"]

                        best_val_loss_str = f"{best_val_loss:.5f}".replace(".", "-")
                        
                        now = datetime.now()
                        ckpt_filename = f"{args.checkpoint_path}/ckpt_"
                        ckpt_filename += f"{now.strftime("%Y%m%d")}_"
                        ckpt_filename += f"{str(time.time()).split(".")[0]}_"
                        ckpt_filename += f"{current_step}_"
                        ckpt_filename += f"{best_val_loss_str}.pt"
                        torch.save(model.state_dict(), ckpt_filename)
    

                if batch != 0: # resetting the running loss
                    metrics["train/loss"] = torch.tensor(0.0, device=DEVICE, dtype=torch.float16)

    wandb.finish()


def val(dataloader, model, device, dm):
    max_batches = len(dataloader)
    h_loss = torch.tensor(0.0,  device=device, dtype=torch.float16)

    compute_acc, compute_precision, compute_recall, compute_f1 = get_metrics(True, device)
    h_acc = 0
    h_precision = 0
    h_recall = 0
    h_f1  = 0
    
    model.eval()
    for samples, labels in dataloader:
        # move data to GPU
        samples = samples.type(torch.float32).to(device)
        labels = labels.type(torch.float32).to(device)

        samples, timesteps = random_forward_steps(dm, int(dm_params["timestep_respacing"]), samples, device=device)

        samples = samples.type(torch.float32)
        timesteps = timesteps.type(torch.float32)


        # inference
        out = model(samples, timesteps)

        # loss
        out = out.reshape(-1)

        epsilon = 0.1
        noise_weights = torch.log((timesteps+epsilon)/1000) * -1
        noise_weights -= noise_weights.min()
        noise_weights /= noise_weights.max()

        label_weight = (labels - 1) * -2

        weight = noise_weights * (label_weight + labels)
        loss = torch.nn.functional.binary_cross_entropy(out, labels, weight=weight)
        
        # stats
        h_loss += loss.detach()
        compute_acc.update(out, labels)
        compute_precision.update(out, labels)
        compute_recall.update(out, labels)
        compute_f1.update(out, labels)

    h_loss /= max_batches
    h_acc = compute_acc.compute().item()
    h_precision = compute_acc.compute().item()
    h_recall = compute_acc.compute().item()
    h_f1 = compute_acc.compute().item()

    return h_loss.item(), h_acc, h_precision, h_recall, h_f1
    
if __name__ == "__main__":
    main()
