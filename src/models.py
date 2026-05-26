import torch
import torch.nn as nn


class SmallEyeCNN(nn.Module):
    """Покадровый классификатор глаза 24x24 в оттенках серого -> открыт/закрыт, ~100k параметров."""

    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def get_features(self, x):
        return self.features(x).flatten(1)

    def forward(self, x):
        return self.classifier(self.features(x))


class CNNLSTM(nn.Module):
    # SmallEyeCNN заморожен, сверху BiLSTM (hidden*2 из за bidirectional)
    def __init__(self, backbone, hidden=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.lstm = nn.LSTM(128, hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, T, 1, EYE_SIZE, EYE_SIZE)
        b, t, c, hh, ww = x.shape
        flat = x.reshape(b * t, c, hh, ww)
        # backbone уже обученая тк grad ненужен
        with torch.no_grad():
            feats = self.backbone.get_features(flat)
        feats = feats.reshape(b, t, -1)
        _, (h_n, _) = self.lstm(feats)
        h = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.classifier(h)
