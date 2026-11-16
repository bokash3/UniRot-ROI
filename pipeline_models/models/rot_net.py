import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class RotResNet18(nn.Module):
    """ResNet18 backbone for rotation with optional cls/reg heads."""
    def __init__(self, mode: str = "both", num_bins: int = 36, grayscale: bool = True, pretrained: bool = True):
        super().__init__()
        assert mode in {"cls", "reg", "both"}
        self.mode = mode
        self.num_bins = num_bins

        # Load backbone
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        backbone = models.resnet18(weights=weights)

        # Adapt first conv for 1-channel masks (mask-only input)
        if grayscale:
            old_conv = backbone.conv1  # shape: (64, 3, 7, 7) if pretrained/standard
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )

            with torch.no_grad():
                # Convert pretrained RGB filters to grayscale by averaging channels
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                if old_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)

            backbone.conv1 = new_conv

        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Heads
        if mode in {"cls", "both"}:
            self.head_cls = nn.Linear(feat_dim, num_bins)
        else:
            self.head_cls = None

        if mode in {"reg", "both"}:
            self.head_reg = nn.Linear(feat_dim, 2)  # (cos, sin)
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
    def vec_to_deg(vec, eps: float = 1e-8):
        # Normalize vector for stable atan2
        v = vec / (vec.norm(dim=1, keepdim=True) + eps)
        angle = torch.atan2(v[:, 1], v[:, 0]) * 180.0 / math.pi
        angle = (angle + 360.0) % 360.0
        return angle

    @staticmethod
    def fuse_both(logits, vec, num_bins: int, alpha: float = 0.5):
        # Fuse classification and regression in a circular-safe way
        reg_v = F.normalize(vec, dim=1)

        bins = logits.argmax(dim=1)
        bin_w = 360.0 / num_bins
        cls_deg = (bins.float() + 0.5) * bin_w
        rad = cls_deg * math.pi / 180.0
        cls_v = torch.stack([torch.cos(rad), torch.sin(rad)], dim=1)

        fused_v = F.normalize(alpha * reg_v + (1.0 - alpha) * cls_v, dim=1)
        return RotResNet18.vec_to_deg(fused_v)
