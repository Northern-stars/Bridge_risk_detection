from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleFuse(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.skip_proj = ConvNormAct(skip_channels, out_channels, kernel_size=1)
        self.conv1 = ConvNormAct(in_channels + out_channels, out_channels)
        self.conv2 = ConvNormAct(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.skip_proj(skip)
        return self.conv2(self.conv1(torch.cat([x, skip], dim=1)))


class FeaturePyramid(nn.Module):
    def __init__(
        self,
        *,
        in_channels: tuple[int, int, int, int],
        pyramid_channels: int = 128,
    ) -> None:
        super().__init__()
        self.lateral1 = nn.Conv2d(in_channels[0], pyramid_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(in_channels[1], pyramid_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels[2], pyramid_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels[3], pyramid_channels, kernel_size=1)

        self.smooth1 = ConvNormAct(pyramid_channels, pyramid_channels)
        self.smooth2 = ConvNormAct(pyramid_channels, pyramid_channels)
        self.smooth3 = ConvNormAct(pyramid_channels, pyramid_channels)
        self.smooth4 = ConvNormAct(pyramid_channels, pyramid_channels)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p4 = self.lateral4(x4)
        p3 = self.lateral3(x3) + F.interpolate(p4, size=x3.shape[-2:], mode="nearest")
        p2 = self.lateral2(x2) + F.interpolate(p3, size=x2.shape[-2:], mode="nearest")
        p1 = self.lateral1(x1) + F.interpolate(p2, size=x1.shape[-2:], mode="nearest")

        return (
            self.smooth1(p1),
            self.smooth2(p2),
            self.smooth3(p3),
            self.smooth4(p4),
        )


class FeatureProjection(nn.Module):
    def __init__(
        self,
        *,
        in_channels: tuple[int, int, int, int],
        out_channels: int,
    ) -> None:
        super().__init__()
        self.project1 = ConvNormAct(in_channels[0], out_channels, kernel_size=1)
        self.project2 = ConvNormAct(in_channels[1], out_channels, kernel_size=1)
        self.project3 = ConvNormAct(in_channels[2], out_channels, kernel_size=1)
        self.project4 = ConvNormAct(in_channels[3], out_channels, kernel_size=1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.project1(x1),
            self.project2(x2),
            self.project3(x3),
            self.project4(x4),
        )


class TransformerMaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        decoder_channels: int,
        num_queries: int,
        num_heads: int,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.memory_proj = nn.Conv2d(in_channels, decoder_channels, kernel_size=1)
        self.query_embed = nn.Embedding(num_queries, decoder_channels)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_channels,
            nhead=num_heads,
            dim_feedforward=decoder_channels * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.query_to_feature = nn.Linear(decoder_channels, decoder_channels)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        memory = self.memory_proj(feature)
        batch_size, channels, height, width = memory.shape
        memory_tokens = memory.flatten(2).transpose(1, 2)
        query_tokens = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        decoded_queries = self.decoder(query_tokens, memory_tokens)
        query_context = self.query_to_feature(decoded_queries).mean(dim=1)
        query_context = query_context.view(batch_size, channels, 1, 1)
        return memory + query_context


class BoundaryGuide(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.boundary_head = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        boundary_logits = self.boundary_head(features)
        boundary_gate = self.gate(boundary_logits.sigmoid())
        guided_features = features * (1.0 + boundary_gate)
        return guided_features, boundary_logits


class ResNet50TransformerBoundarySegmentation(nn.Module):
    """Dilated ResNet50 encoder with optional FPN, transformer decoder, and boundary guide."""

    def __init__(
        self,
        num_classes: int,
        *,
        pretrained: bool = False,
        decoder_channels: int = 128,
        num_heads: int = 4,
        num_queries: int = 64,
        use_fpn: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        encoder = resnet50(
            weights=weights,
            replace_stride_with_dilation=(False, True, True),
        )
        encoder_channels = (256, 512, 1024, 2048)

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.feature_neck = (
            FeaturePyramid(in_channels=encoder_channels, pyramid_channels=decoder_channels)
            if use_fpn
            else FeatureProjection(in_channels=encoder_channels, out_channels=decoder_channels)
        )
        self.transformer_decoder = TransformerMaskDecoder(
            in_channels=decoder_channels,
            decoder_channels=decoder_channels,
            num_queries=num_queries,
            num_heads=num_heads,
        )
        self.up4 = UpsampleFuse(decoder_channels, decoder_channels, 128)
        self.up3 = UpsampleFuse(128, decoder_channels, 96)
        self.up2 = UpsampleFuse(96, decoder_channels, 64)
        self.up1 = UpsampleFuse(64, 64, 64)
        self.refine = nn.Sequential(
            ConvNormAct(64, 64),
            ConvNormAct(64, 64),
        )
        self.boundary_guide = BoundaryGuide(64)
        self.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]

        x0 = self.stem(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        p1, p2, p3, p4 = self.feature_neck(x1, x2, x3, x4)

        features = self.transformer_decoder(p4)
        features = self.up4(features, p3)
        features = self.up3(features, p2)
        features = self.up2(features, p1)
        features = self.up1(features, x0)
        features = F.interpolate(features, size=input_size, mode="bilinear", align_corners=False)
        features = self.refine(features)

        guided_features, boundary_logits = self.boundary_guide(features)
        seg_logits = self.segmentation_head(guided_features)

        return {
            "seg_logits": seg_logits,
            "boundary_logits": boundary_logits,
        }


def create_model(
    num_classes: int,
    *,
    pretrained: bool = False,
    decoder_channels: int = 128,
    num_heads: int = 4,
    num_queries: int = 64,
    use_fpn: bool = True,
) -> ResNet50TransformerBoundarySegmentation:
    return ResNet50TransformerBoundarySegmentation(
        num_classes=num_classes,
        pretrained=pretrained,
        decoder_channels=decoder_channels,
        num_heads=num_heads,
        num_queries=num_queries,
        use_fpn=use_fpn,
    )
