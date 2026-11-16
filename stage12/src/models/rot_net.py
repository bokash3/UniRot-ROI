import torch
import torch.nn as nn
import torchvision.models as models
import math

class RotResNet18(nn.Module):
    """ResNet18 backbone for rotation with optional cls/reg heads."""
    def __init__(self, mode: str = "both", num_bins: int = 36, grayscale: bool = True):
        super().__init__()
        assert mode in {"cls", "reg", "both"}
        self.mode = mode
        self.num_bins = num_bins
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if grayscale:
            # Replace first conv to accept 1-channel input
            backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = backbone
        feat_dim = backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        if mode in {"cls", "both"}:
            self.head_cls = nn.Linear(feat_dim, num_bins)
        else:
            self.head_cls = None

        if mode in {"reg", "both"}:
            # Predict 2D unit vector (cos, sin) to avoid angle wrap
            self.head_reg = nn.Linear(feat_dim, 2)
        else:
            self.head_reg = None

    def forward(self, x):
        f = self.backbone(x)
        logits, vec = None, None
        if self.head_cls is not None:
            logits = self.head_cls(f)
        if self.head_reg is not None:
            vec = self.head_reg(f)
        return logits, vec

    @staticmethod
    def vec_to_deg(vec):
        # vec: (N,2)
        cos = vec[:,0]
        sin = vec[:,1]
        angle = torch.atan2(sin, cos) * 180.0 / math.pi
        angle = (angle + 360.0) % 360.0
        return angle
