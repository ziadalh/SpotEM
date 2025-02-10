import torch
import torch.nn as nn


class EfficientNet_b0(nn.Module):
    def __init__(self, dim=1280, pretrained=True):
        super().__init__()
        effnet = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_efficientnet_b0",
            pretrained=pretrained,
        )
        features = effnet.features
        if dim != 1280:
            features.conv = nn.Conv2d(
                320, dim, kernel_size=(1, 1), stride=(1, 1), bias=False
            )
            features.bn = nn.BatchNorm2d(dim, eps=0.001, momentum=0.01)

        self.main = nn.Sequential(
            effnet.stem,
            effnet.layers,
            features,
            effnet.classifier.pooling,
            effnet.classifier.squeeze,
        )

    def forward(self, x):
        return self.main(x)
