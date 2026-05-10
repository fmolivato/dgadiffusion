import torch
import argparse
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, Accuracy, Precision, Recall, F1Score
from guided_diffusion.guided_diffusion.respace import SpacedDiffusion
import random

def get_metrics(to_device=False, device="cuda"):
    compute_acc = BinaryAccuracy()
    compute_precision = BinaryPrecision()
    compute_recall = BinaryRecall()
    compute_f1 = BinaryF1Score()

    if to_device:
        compute_acc = compute_acc.to(device)
        compute_precision = compute_precision.to(device)
        compute_recall = compute_recall.to(device)
        compute_f1 = compute_f1.to(device)

    return compute_acc, compute_precision, compute_recall, compute_f1

def get_multiclass_metrics(task, num_classes, to_device=False, device="cuda"):
    compute_acc = Accuracy(task, num_classes=num_classes)
    compute_precision = Precision(task, num_classes=num_classes)
    compute_recall = Recall(task, num_classes=num_classes)
    compute_f1 = F1Score(task, num_classes=num_classes)

    if to_device:
        compute_acc = compute_acc.to(device)
        compute_precision = compute_precision.to(device)
        compute_recall = compute_recall.to(device)
        compute_f1 = compute_f1.to(device)

    return compute_acc, compute_precision, compute_recall, compute_f1

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
    

def torch_batch_timesteps(dm, x_start, same_batch_noise = False, device=None):
    """Perform the forward process of the diffusion model for a specific time 't'.

    Args:
        x_start (np.array): batch of images to which the noise is applied.
        same_batch_noise (bool): choose wether tho apply the same noise to all images in the batch or different one.

    Returns:
        np.array: batch of images corrupted by the noise (diffusion forward)
    """

    batch, height, width, channels = x_start.shape
    noise = torch.randn(*x_start.shape) # zero mean unit variance coherent with imagenet sample normalization
    #noise = torch.clip(noise, 0, 1)

    # ablation?
    if same_batch_noise:
        alphas = dm.sqrt_alphas_cumprod
        perm = torch.randperm(1000)
        alpha_index = perm[0].numpy()
        alpha = alphas[alpha_index]
    else:
        perm = torch.randperm(1000)
        alpha_index = perm[:batch].numpy()
        alpha = dm.sqrt_alphas_cumprod[alpha_index]
        alpha = alpha.reshape(batch, 1, 1, 1)

    alpha = torch.tensor(alpha)
    alpha_index = torch.tensor(alpha_index)
    if device is not None:
        alpha = alpha.to(device)
        noise = noise.to(device)
        alpha_index = alpha_index.to(device)

    out = x_start * alpha + (1 - alpha) * noise

    return out, alpha_index

def torch_batch_timesteps_multinomial(dm, x_start, same_batch_noise = False, noise_step=None, device=None):
    """Perform the forward process of the diffusion model for a specific time 't'.

    Args:
        x_start (np.array): batch of images to which the noise is applied.
        same_batch_noise (bool): choose wether tho apply the same noise to all images in the batch or different one.
        noise_step (int): define the exact noise index to apply. only if 'same_batch_noise' is True.
        device (strign): device where to move data e.g. 'cpu'.

    Returns:
        torch.tensor: batch of images corrupted by the noise (diffusion forward)
        torch.tensor: indeces of the noise levels used in the batch
    """

    batch, height, width, channels = x_start.shape
    noise = torch.randn(*x_start.shape) # zero mean unit variance coherent with imagenet sample normalization

    # ablation?
    weight = torch.tensor(list(range(1000, 0, -1)), dtype=torch.float)
    if same_batch_noise:
        alphas = dm.sqrt_alphas_cumprod
        alpha_index = torch.multinomial(weight, 1)[0] if noise_step is None else torch.tensor(int(noise_step))
        alpha_index = torch.stack([alpha_index for _ in range(batch)]).numpy()
        alpha = alphas[alpha_index]
    else:
        alpha_index = torch.multinomial(weight, batch).numpy()
        alpha = dm.sqrt_alphas_cumprod[alpha_index]
    
    alpha = alpha.reshape(batch, 1, 1, 1)
    alpha = torch.tensor(alpha)
    alpha_index = torch.tensor(alpha_index)
    if device is not None:
        alpha = alpha.to(device)
        noise = noise.to(device)
        alpha_index = alpha_index.to(device)

    out = x_start * alpha + (1 - alpha) * noise

    return out, alpha_index


def   random_forward_steps(dm: SpacedDiffusion, max_steps : int, samples : torch.tensor, same_noise : bool =False, device: str = None) -> torch.tensor:
    """Apply noise (forward process) to a batch of samples

    Args:
        dm (SpacedDiffusion): diffusion model
        max_steps (SpacedDiffusion): max numbers of rescaled steps (if rescaled) used by the dm
        samples (torch.tensor): batch of data to which apply the forward steps
        same_noise (bool): when True use the same noise step for all samples in the batch else different noise levels
        device (str): device where to create the tensors

    Rerturns:
        torch.tensor: batch of samples with additional noise
    """

    # choose random steps
    B, _, _, _ = samples.shape
    dummy_tensor = torch.zeros((B,))
    timesteps = None

    if same_noise:
        rnd_int = random.randint(0, max_steps)
        timesteps = torch.randint_like(dummy_tensor, rnd_int, rnd_int+1, dtype=torch.long, device=device)
    else:
        timesteps = torch.randint_like(dummy_tensor, 0, max_steps, dtype=torch.long, device=device)

    # apply noise to the batch
    samples = dm.q_sample(samples, timesteps)

    return samples, timesteps