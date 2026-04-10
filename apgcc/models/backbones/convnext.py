"""
ConvNeXt backbone wrapper for APGCC.
Uses torchvision's ConvNeXt (available in torchvision >= 0.13).
Variants: convnext_tiny, convnext_small, convnext_base, convnext_large
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models


# Feature map channel dimensions for each ConvNeXt variant
# Outputs from stages 1-4: [96, 192, 384, 768] for tiny/small, [128,256,512,1024] for base/large
_CONVNEXT_OUTPLANES = {
    'convnext_tiny':   [96,  192,  384,  768],
    'convnext_small':  [96,  192,  384,  768],
    'convnext_base':   [128, 256,  512,  1024],
    'convnext_large':  [192, 384,  768,  1536],
}


class Base_ConvNeXt(nn.Module):
    """
    ConvNeXt encoder that outputs 4 feature maps at strides 4, 8, 16, 32
    matching the interface expected by APGCC's decoder (same as VGG/ResNet encoders).
    """
    def __init__(self, name: str, last_pool: bool = False, num_channels: int = 256, **kwargs):
        super().__init__()
        print(f"### ConvNeXt: name={name}, last_pool={last_pool}")

        weights_map = {
            'convnext_tiny':  tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
            'convnext_small': tv_models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
            'convnext_base':  tv_models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
            'convnext_large': tv_models.ConvNeXt_Large_Weights.IMAGENET1K_V1,
        }
        if name not in weights_map:
            raise ValueError(f"Unknown ConvNeXt variant '{name}'. "
                             f"Choose from: {list(weights_map.keys())}")

        backbone = getattr(tv_models, name)(weights=weights_map[name])

        # ConvNeXt features layout (torchvision):
        #   features[0] = stem          (stride 4)
        #   features[1] = stage 1 (downsample + blocks)
        #   features[2] = stage 2 (stride 8)
        #   features[3] = stage 2 blocks
        #   features[4] = stage 3 (stride 16)
        #   features[5] = stage 3 blocks
        #   features[6] = stage 4 (stride 32)
        #   features[7] = stage 4 blocks
        f = backbone.features
        self.body1 = nn.Sequential(f[0], f[1])          # stride 4
        self.body2 = nn.Sequential(f[2], f[3])          # stride 8
        self.body3 = nn.Sequential(f[4], f[5])          # stride 16
        if last_pool:
            self.body4 = nn.Sequential(f[6], f[7],
                                       nn.MaxPool2d(2))  # stride 64 (matches last_pool=True)
        else:
            self.body4 = nn.Sequential(f[6], f[7])      # stride 32

        self._outplanes = _CONVNEXT_OUTPLANES[name]
        self.num_channels = num_channels
        self.last_pool = last_pool

    def get_outplanes(self):
        return list(self._outplanes)

    def forward(self, x):
        out = []
        for layer in [self.body1, self.body2, self.body3, self.body4]:
            x = layer(x)
            out.append(x)
        return out  # [stride4, stride8, stride16, stride32]
