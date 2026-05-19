# src/models/unet.py
"""U-Net decoder with skip connections. Encoder-agnostic:
accepts any encoder exposing .stage_channels and .forward() -> list[Tensor].
"""
import torch
import torch.nn as nn


class UNet(nn.Module):
    """U-Net: encoder (pluggable) + decoder with skip connections.

    Input:  (B, in_channels, 256, 256)
    Output: (B, out_channels, 256, 256) logits
    """

    def __init__(self, encoder, out_channels=1):
        super().__init__()
        self.encoder = encoder
        ch = encoder.stage_channels  # e.g. [64, 128, 256, 512]

        # Bottleneck: deepest encoder output -> same spatial, wider channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch[-1], ch[-1] * 2, 3, padding=1),
            nn.BatchNorm2d(ch[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[-1] * 2, ch[-1] * 2, 3, padding=1),
            nn.BatchNorm2d(ch[-1] * 2),
            nn.ReLU(inplace=True),
        )

        # Decoder stages (reversed): upsample + concat skip + conv block
        # Stage i takes bottleneck/prev output, upsamples, concats with encoder feature i
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        in_ch = ch[-1] * 2  # bottleneck output channels
        for skip_ch in reversed(ch):
            self.up_convs.append(
                nn.ConvTranspose2d(in_ch, skip_ch, kernel_size=2, stride=2)
            )
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(skip_ch * 2, skip_ch, 3, padding=1),  # concat doubles channels
                nn.BatchNorm2d(skip_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(skip_ch, skip_ch, 3, padding=1),
                nn.BatchNorm2d(skip_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = skip_ch

        # Final 1x1 conv to output logits
        self.head = nn.Conv2d(ch[0], out_channels, 1)

    def forward(self, x):
        # Encode
        features = self.encoder(x)              # [s1, s2, s3, s4]

        # Bottleneck (pool the last encoder feature, then process)
        x = nn.functional.max_pool2d(features[-1], 2)
        x = self.bottleneck(x)                  # (B, ch[-1]*2, H/16, W/16)

        # Decode with skip connections (deepest to shallowest)
        for i, (up, dec) in enumerate(zip(self.up_convs, self.dec_blocks)):
            skip = features[-(i + 1)]           # matching encoder feature
            x = up(x)                           # upsample
            x = torch.cat([x, skip], dim=1)     # concat along channels
            x = dec(x)                          # refine

        return self.head(x)