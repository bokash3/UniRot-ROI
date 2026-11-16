import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Model: Palm ROI-Net
# -----------------------------
class PalmROINet(nn.Module):
    """
    Lightweight ROI regression network:
    - Backbone: MobileNetV2 (1-channel input).
    - Output: 3D vector in [0,1]: (cx, cy, size).
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        # Load MobileNetV2 backbone
        # Note: weights=None to avoid downloading; set manually if you have cached weights.
        self.backbone = models.mobilenet_v2(weights=None)

        # Modify first conv layer to accept 1-channel input instead of 3.
        first_conv = self.backbone.features[0][0]  # Conv2d
        new_first_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        # Optionally, you can average weights of original conv over the 3 RGB channels.
        with torch.no_grad():
            if pretrained and first_conv.weight.shape[1] == 3:
                # Average RGB channels to initialize the single channel
                w = first_conv.weight.mean(dim=1, keepdim=True)
                new_first_conv.weight.copy_(w)
            else:
                # Random init (default Kaiming by PyTorch)
                pass

        self.backbone.features[0][0] = new_first_conv

        # Replace classifier head: MobileNetV2 classifier is usually (Dropout + Linear)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

        # Final activation: we want outputs in [0,1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B,1,H,W), normalized in [0,1].

        Returns:
            Tensor of shape (B,3) in [0,1]: (cx, cy, size).
        """
        out = self.backbone(x)  # (B,3) raw
        out = self.sigmoid(out)
        return out
