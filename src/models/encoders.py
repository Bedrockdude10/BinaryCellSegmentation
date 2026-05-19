# src/models/encoders.py
"""Encoder backbones for U-Net. All expose the same interface:
    - forward(x) -> list[Tensor]   (one feature map per stage, low-to-high level)
    - stage_channels -> list[int]  (channel count at each stage)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _vgg_block(in_ch, out_ch, num_convs=2):
    """VGG-style block: repeated 3x3 Conv -> BN -> ReLU."""
    layers = []
    for i in range(num_convs):
        layers += [
            nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


class VGGEncoder(nn.Module):
    """VGG-style encoder: 3x3 convs, pool + double channels.

    4 stages: 64 -> 128 -> 256 -> 512
    Input: (B, 3, 256, 256)
    Outputs: [(B,64,256,256), (B,128,128,128), (B,256,64,64), (B,512,32,32)]
    """

    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        c = base_channels
        self._stage_channels = [c, c * 2, c * 4, c * 8]

        self.stage1 = _vgg_block(in_channels, c, num_convs=2)
        self.stage2 = _vgg_block(c, c * 2, num_convs=2)
        self.stage3 = _vgg_block(c * 2, c * 4, num_convs=3)
        self.stage4 = _vgg_block(c * 4, c * 8, num_convs=3)
        self.pool = nn.MaxPool2d(2, stride=2)

    @property
    def stage_channels(self):
        return list(self._stage_channels)

    def forward(self, x):
        features = []
        x = self.stage1(x)
        features.append(x)          # (B, 64, 256, 256)
        x = self.stage2(self.pool(x))
        features.append(x)          # (B, 128, 128, 128)
        x = self.stage3(self.pool(x))
        features.append(x)          # (B, 256, 64, 64)
        x = self.stage4(self.pool(x))
        features.append(x)          # (B, 512, 32, 32)
        return features


class ResNetEncoder(nn.Module):
    """ResNet-style encoder with residual (skip) connections.

    Uses a stem conv to get to base_channels, then 4 stages with
    BasicBlock residuals. Downsampling via stride-2 conv in first
    block of each stage (except stage 1).

    4 stages: 64 -> 128 -> 256 -> 512
    Input: (B, 3, 256, 256)
    Outputs: [(B,64,256,256), (B,128,128,128), (B,256,64,64), (B,512,32,32)]
    """

    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        c = base_channels
        self._stage_channels = [c, c * 2, c * 4, c * 8]

        # Stem: 3x3 conv (no spatial reduction, unlike standard ResNet)
        # Keeps 256x256 so encoder features match VGG spatial dims
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=1, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # stage1 at full res, stages 2-4 downsample via stride-2 first block
        self.stage1 = self._make_stage(c, c, blocks=2, stride=1)
        self.stage2 = self._make_stage(c, c * 2, blocks=2, stride=2)
        self.stage3 = self._make_stage(c * 2, c * 4, blocks=2, stride=2)
        self.stage4 = self._make_stage(c * 4, c * 8, blocks=2, stride=2)

    @property
    def stage_channels(self):
        return list(self._stage_channels)

    @staticmethod
    def _make_stage(in_ch, out_ch, blocks, stride):
        layers = [_ResBasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(_ResBasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        x = self.stem(x)            # (B, 64, 256, 256)
        x = self.stage1(x)
        features.append(x)          # (B, 64, 256, 256)
        x = self.stage2(x)
        features.append(x)          # (B, 128, 128, 128)
        x = self.stage3(x)
        features.append(x)          # (B, 256, 64, 64)
        x = self.stage4(x)
        features.append(x)          # (B, 512, 32, 32)
        return features


class _ResBasicBlock(nn.Module):
    """Pre-activation style residual block: BN -> ReLU -> Conv."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class SwinEncoder(nn.Module):
    """Swin Transformer encoder wrapping timm's Swin-T.

    Same interface as VGG/ResNet encoders:
        forward(x) -> list of 4 feature maps
        stage_channels -> [96, 192, 384, 768]

    Swin-T with patch_size=4 natively produces features at spatial
    resolutions H/4, H/8, H/16, H/32 (i.e. 64, 32, 16, 8 for 256x256).
    The CNN encoders produce features at H, H/2, H/4, H/8 (256, 128, 64, 32).

    To bridge this gap, each stage's output is bilinearly upsampled 4x so
    the decoder receives features at 256, 128, 64, 32 — matching the CNN
    encoders exactly. This is a standard adaptation when plugging
    transformer backbones into CNN-style decoders.
    """

    # Target spatial sizes for 256x256 input, matching VGG/ResNet
    _TARGET_SIZES = [256, 128, 64, 32]

    def __init__(self, in_channels=3, embed_dim=96, pretrained=False):
        super().__init__()
        import timm

        # Create Swin-T backbone via timm with feature extraction
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            in_chans=in_channels,
            img_size=256,
            features_only=True,
        )

        # Swin-T stage channels: [96, 192, 384, 768]
        self._stage_channels = self.backbone.feature_info.channels()

    @property
    def stage_channels(self):
        return list(self._stage_channels)

    def forward(self, x):
        # timm's Swin may return (B, H, W, C) channels-last tensors.
        # We normalize everything to (B, C, H, W) before upsampling.
        raw_features = self.backbone(x)

        features = []
        for feat, expected_ch, target_size in zip(
            raw_features, self._stage_channels, self._TARGET_SIZES
        ):
            # If channels-last: shape is (B, H, W, C) and dim[-1] == expected_ch
            if feat.shape[-1] == expected_ch and feat.shape[1] != expected_ch:
                feat = feat.permute(0, 3, 1, 2).contiguous()

            # Upsample to match CNN encoder spatial resolutions
            if feat.shape[-1] != target_size:
                feat = F.interpolate(
                    feat, size=(target_size, target_size),
                    mode="bilinear", align_corners=False,
                )
            features.append(feat)

        return features