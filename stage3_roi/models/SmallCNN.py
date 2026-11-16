import torch.nn as nn

class SmallCNNBaseline(nn.Module):
    """
    Simple CNN baseline for ROI regression.
    """
    def __init__(self, img_size: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        feat_hw = img_size // (2 ** 4)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * feat_hw * feat_hw, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 3),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return self.sigmoid(x)

