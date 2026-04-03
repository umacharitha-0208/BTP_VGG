# adapted from: https://github.com/Parskatt/DeDoDe/blob/main/DeDoDe/encoder.py and https://github.com/Parskatt/DeDoDe/blob/main/DeDoDe/decoder.py

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torchvision.models as tvm

from ripe.utils import get_pylogger

log = get_pylogger(__name__)


class Decoder(nn.Module):
    def __init__(self, layers, *args, super_resolution=False, num_prototypes=1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.scales = self.layers.keys()
        self.super_resolution = super_resolution
        self.num_prototypes = num_prototypes

    def forward(self, features, context=None, scale=None):
        if context is not None:
            features = torch.cat((features, context), dim=1)
        stuff = self.layers[scale](features)
        logits, context = (
            stuff[:, : self.num_prototypes],
            stuff[:, self.num_prototypes :],
        )
        return logits, context


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=True,
        kernel_size=5,
        hidden_blocks=5,
        residual=False,
    ):
        super().__init__()
        self.block1 = self.create_block(
            in_dim,
            hidden_dim,
            dw=False,
            kernel_size=1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        self.residual = residual

    def create_block(
        self,
        in_dim,
        out_dim,
        dw=True,
        kernel_size=5,
        bias=True,
        norm_type=nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert out_dim % in_dim == 0, "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = norm_type(out_dim) if norm_type is nn.BatchNorm2d else norm_type(num_channels=out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, feats):
        b, c, hs, ws = feats.shape
        x0 = self.block1(feats)
        x = self.hidden_blocks(x0)
        if self.residual:
            x = (x + x0) / 1.4
        x = self.out_conv(x)
        return x


class VGG19(nn.Module):
    def __init__(self, pretrained=False, num_input_channels=3, use_gradient_checkpointing=False, freeze_layers=0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        self.use_gradient_checkpointing = use_gradient_checkpointing
        # Maxpool layers: 6, 13, 26, 39

        if num_input_channels != 3:
            log.info(f"Changing input channels from 3 to {num_input_channels}")
            self.layers[0] = nn.Conv2d(num_input_channels, 64, 3, 1, 1)
        
        # Freeze early layers for memory optimization
        if freeze_layers > 0:
            log.info(f"Freezing first {freeze_layers} layers of VGG19 for memory optimization")
            for i, layer in enumerate(self.layers):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

    def get_dim_layers(self):
        return [64, 128, 256, 512]

    def _forward_block(self, x, start_idx, end_idx):
        """Forward pass through a block of layers."""
        for layer in self.layers[start_idx:end_idx]:
            x = layer(x)
        return x

    def forward(self, x, **kwargs):
        feats = []
        sizes = []
        
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            # Split into 4 blocks at maxpool boundaries: [0-6], [7-13], [14-26], [27-40]
            block_boundaries = [0, 7, 14, 27, 40]
            
            for i in range(len(block_boundaries) - 1):
                start, end = block_boundaries[i], block_boundaries[i + 1]
                
                # Store features before maxpool
                if i > 0:  # Skip first iteration (no maxpool before first block)
                    feats.append(x)
                    sizes.append(x.shape[-2:])
                
                # Use checkpointing for this block
                x = checkpoint.checkpoint(
                    self._forward_block,
                    x,
                    start,
                    end,
                    use_reentrant=False
                )
        else:
            # Standard forward pass
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats.append(x)
                    sizes.append(x.shape[-2:])
                x = layer(x)
        
        return feats, sizes


class VGG(nn.Module):
    def __init__(self, size="19", pretrained=False) -> None:
        super().__init__()
        if size == "11":
            self.layers = nn.ModuleList(tvm.vgg11_bn(pretrained=pretrained).features[:22])
        elif size == "13":
            self.layers = nn.ModuleList(tvm.vgg13_bn(pretrained=pretrained).features[:28])
        elif size == "19":
            self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        # Maxpool layers: 6, 13, 26, 39

    def forward(self, x, **kwargs):
        feats = []
        sizes = []
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                feats.append(x)
                sizes.append(x.shape[-2:])
            x = layer(x)
        return feats, sizes
