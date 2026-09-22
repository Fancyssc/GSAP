"""Reuse baseline segmentation; replace only its eight attention modules."""
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from mmseg.registry import MODELS
from .spiking_baseline import spiking_baseline

__all__ = ['spiking_gsap']

class GSAPAttention(nn.Module):
    def __init__(self, dim, feature_size=14, backend='cupy'):
        super().__init__()
        assert dim >= 1 and feature_size >= 1
        self.feature_size = feature_size
        kernel_size = 2 * feature_size - 1
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.feature_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.feature_bn = nn.BatchNorm2d(dim)
        self.feature_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.local_bn = nn.BatchNorm2d(dim)
        self.row_conv = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), padding=(0, feature_size - 1), groups=dim, bias=False)
        self.row_bn = nn.BatchNorm2d(dim)
        self.row_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.column_conv = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=(feature_size - 1, 0), groups=dim, bias=False)
        self.column_bn = nn.BatchNorm2d(dim)
        self.gate_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.gate_bn = nn.BatchNorm2d(dim)
        self.gate_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=backend)
        self.attn_bn = nn.BatchNorm2d(dim)
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=backend)
        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        spike = self.proj_lif(x).flatten(0, 1)
        local = self.local_bn(self.local_conv(spike))
        gate = self.gate_bn(self.gate_conv(spike)).reshape(T, B, C, H, W)
        gate = self.gate_lif(gate).flatten(0, 1)
        feature = self.feature_bn(self.feature_conv(spike)).reshape(T, B, C, H, W)
        feature = self.feature_lif(feature).flatten(0, 1)

        message = self.row_bn(self.row_conv(feature)).reshape(T, B, C, H, W)
        message = self.row_lif(message + feature.reshape(T, B, C, H, W)).flatten(0, 1)
        message = self.column_bn(self.column_conv(message))
        x = self.attn_bn(spike + local + message * gate).reshape(T, B, C, H, W)
        x = self.attn_lif(x).flatten(0, 1)
        return self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W)


@MODELS.register_module()
def spiking_gsap(pretrained=None, **kwargs):
    # Fixed kernels support whole images; range does not grow with the image.
    feature_size = (max(kwargs.get('img_size_h', 224),
                        kwargs.get('img_size_w', 224)) + 15) // 16
    model = spiking_baseline(pretrained=pretrained, **kwargs)
    for block in [*model.block3, *model.block4]:
        name = 'attn'
        dim = block.mlp.mlp1_conv.in_channels
        setattr(block, name, GSAPAttention(dim, feature_size))
    return model
