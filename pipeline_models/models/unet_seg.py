import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetSeg(nn.Module):
    """U-Net wrapper using SMP with configurable encoder."""
    def __init__(self, encoder_name: str = "resnet34", encoder_weights: str = "imagenet"):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1
        )

    def forward(self, x):
        return self.model(x)
