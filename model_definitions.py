import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import torch.nn.functional as F
import torch.nn as nn
import torch

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, padding=2, kernel_size=5)
        self.pool1 = nn.AvgPool2d((2, 2))

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d((2, 2))

        self.fc1 = nn.Linear(16*749*2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        y = self.conv1(x)
        y = F.relu(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = F.relu(y)
        y = self.pool2(y)
        y = y.flatten(1, -1)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = F.relu(y)
        y = self.fc3(y)
        y = F.relu(y)
        return y

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        dropout_p = 0.3
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            nn.Dropout(dropout_p),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            nn.Dropout(dropout_p),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 751 * 4, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_p),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x, _ = layer(x)
            else:
                x = layer(x)

        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, n_features):
        super(ResidualBlock, self).__init__()
        dropout_p = 0.2
        self.conv_block = nn.Sequential(
            nn.Conv2d(n_features, n_features, 3, 1, 1),
            nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),

            nn.Conv2d(n_features, n_features, 3, 1, 1),
            nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x) + x


class Resnet(nn.Module):
    def __init__(self, in_ch=1, out_ch=64, n_blocks=8):
        super(Resnet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_blocks = n_blocks
        n_features = 64

        features = [
            nn.Conv2d(in_ch, n_features, 4, 2, 1),
            nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True)
        ]

        for i in range(self.n_blocks):
            features += [ResidualBlock(n_features)]

        features += [
            nn.Conv2d(n_features, out_ch, 4, 2, 1),
            nn.ReLU(inplace=True)
        ]

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Linear(self.out_ch * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x