#
#
#

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from guided_diffusion.guided_diffusion.unet import EncoderUNetModel, AttentionPool2d

classifier_params = {
    "image_size": 256,
    "classifier_use_fp16": False,
    "classifier_width": 128,
    "classifier_depth": 2,
    "classifier_attention_resolutions":  "32,16,8",
    "classifier_use_scale_shift_norm": True,
    "classifier_resblock_updown": True,
    "classifier_pool": "attention",
}

def create_classifier(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
):
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=1000,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # backbone
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Identity()

        # new head
        self.lin = nn.Linear(512, 1)
        torch.nn.init.kaiming_uniform_(self.lin.weight)

        # activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)

        x = self.lin(x)
        x = self.sigmoid(x)
        return x
    
class DiscriminatorEncoder(nn.Module):
    def __init__(self, weights_path="./pretrained_weights/256x256_classifier.pt", trainable_backbone=False):
        super().__init__()
        self.trainable_backbone = trainable_backbone

        # backbone
        self.model = create_classifier(**classifier_params)
        self.model.load_state_dict(torch.load(weights_path, weights_only=False))

        # new head
        self.model.out = nn.Identity()
        self.atn = AttentionPool2d(7, 512, 64, 1)

        # activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, timesteps):

        if self.trainable_backbone:
            x = self.model(x, timesteps)
        else:
            with torch.no_grad():
                x = self.model(x, timesteps)

        #x = torch.nn.functional.silu(x)
        x = self.atn(x)
        x = self.sigmoid(x)
        return x
    
class DiscriminatorEncoderLinearHeads(nn.Module):
    def __init__(self, weights_path="./pretrained_weights/256x256_classifier.pt", trainable_backbone=False):
        super().__init__()
        self.trainable_backbone = trainable_backbone

        # backbone
        self.model = create_classifier(**classifier_params)
        self.model.load_state_dict(torch.load(weights_path, weights_only=False))

        # new head
        self.model.out = nn.Identity()
        self.group_norm = nn.GroupNorm(32, 512)

        self.atn = AttentionPool2d(7, 512, 64, 512)
        
        self.lin_1 = nn.Linear(512, 1)
        torch.nn.init.kaiming_uniform_(self.lin_1.weight)

        # activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, timesteps):

        if self.trainable_backbone:
            x = self.model(x, timesteps)
        else:
            with torch.no_grad():
                x = self.model(x, timesteps)

        x = self.group_norm(x)
        x = torch.nn.functional.silu(x)

        x = self.atn(x)     
        x = torch.nn.functional.silu(x)
        
        x = self.lin_1(x)
        x = self.sigmoid(x)
        
        return x

def get_model(model_name="resnet", trainable_backbone=False) -> nn.Module:
    """Factory function to get the right model object

    Args:
        model_name (str): the model name loads specific architectures
        trainable_backbone (bool): wheather to train only the head or also the backbone
    Returns:
        nn.Module: the object of the choosen architecture
    """
    
    match model_name:
        case "resnet":
            model = Discriminator() 
        case "unet_enc_atn":
            model = DiscriminatorEncoder(trainable_backbone=trainable_backbone)
        case "unet_enc_lin":
            model = DiscriminatorEncoderLinearHeads(trainable_backbone=trainable_backbone)
        case _:
            raise Exception(f"There is no '{model_name}' model")

    return model


if __name__ == "__main__":
    #model = create_classifier(**classifier_params)
    model = get_model("unet_enc_atn")
    print(model)

    sample = torch.randn((1, 3, 224, 224))
    timestep = torch.zeros([1])
    out = model(sample, timestep)
    print(out.shape)

