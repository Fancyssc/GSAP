"""UCF101-DVS 2-256 adaptation of GSAP (single hardcoded entrypoint).

Reset neuron state between independent batches with functional.reset_net(model).
"""
import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

__all__ = ['spiking_gsap']


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., backend='cupy'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend=backend)
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp2_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend=backend)
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.mlp1_lif(x)
        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)

        x = self.mlp2_lif(x)
        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, self.c_output, H, W)
        return x


class GSAPAttention(nn.Module):
    def __init__(self, dim, feature_size=8, backend='cupy'):
        super().__init__()
        assert dim >= 1 and feature_size >= 1
        self.feature_size = feature_size
        kernel_size = 2 * feature_size - 1
        self.proj_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend=backend)
        self.feature_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.feature_bn = nn.BatchNorm2d(dim)
        self.feature_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend=backend)
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.local_bn = nn.BatchNorm2d(dim)
        self.row_conv = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), padding=(0, feature_size - 1), groups=dim, bias=False)
        self.row_bn = nn.BatchNorm2d(dim)
        self.row_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend=backend)
        self.column_conv = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=(feature_size - 1, 0), groups=dim, bias=False)
        self.column_bn = nn.BatchNorm2d(dim)
        self.gate_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.gate_bn = nn.BatchNorm2d(dim)
        self.gate_lif = MultiStepLIFNode(tau=1.5, v_threshold=0.5, detach_reset=True, backend=backend)
        self.attn_bn = nn.BatchNorm2d(dim)
        self.attn_lif = MultiStepLIFNode(tau=1.5, v_threshold=0.5, detach_reset=True, backend=backend)
        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        assert H <= self.feature_size and W <= self.feature_size, 'Increase feature_size to cover the full input graph'
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


class SpikingTransformer(nn.Module):
    def __init__(self, dim, mlp_ratio=4., feature_size=8, backend='cupy'):
        super().__init__()
        self.attn = GSAPAttention(dim, feature_size=feature_size, backend=backend)
        self.mlp = MLP(dim, hidden_features=int(dim * mlp_ratio), backend=backend)

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


class SpikingTokenizer(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, embed_dims=256, backend='cupy'):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        if patch_size != (16, 16):
            raise ValueError('UCF101-DVS tokenizer has four stride-2 pools; patch_size must be 16')
        if min(img_size_h, img_size_w) < 1 or embed_dims < 8 or embed_dims % 8:
            raise ValueError('Image dimensions must be positive; embed_dims must be a multiple of 8')
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = (img_size_h + 15) // 16, (img_size_w + 15) // 16
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)

        self.proj1_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend=backend)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj1_conv = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims//4)

        self.proj2_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend=backend)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj2_conv = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims//2)

        self.proj3_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend=backend)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)

        self.proj4_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend=backend)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1)))
        # Match the corrected DVS prototype: LIF -> pool -> Conv -> BN.
        # Read actual sizes instead of hardcoding 64/32/16 for the reshapes.
        for stage in range(1, 5):
            C, H, W = x.shape[1:]
            x = getattr(self, f'proj{stage}_lif')(x.reshape(T, B, C, H, W).contiguous())
            x = getattr(self, f'maxpool{stage}')(x.flatten(0, 1))
            x = getattr(self, f'proj{stage}_conv')(x)
            x = getattr(self, f'proj{stage}_bn')(x)
        H, W = x.shape[-2:]
        return x.reshape(T, B, -1, H, W), (H, W)


class vit_snn(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=16,
                 in_channels=2, num_classes=101, embed_dims=256, num_heads=16,
                 mlp_ratios=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=2, sr_ratios=1, T=16, pretrained_cfg=None,
                 pretrained_cfg_overlay=None, drop_block_rate=None, backend='cupy'):
        super().__init__()
        if not isinstance(depths, int) or depths < 1 or not isinstance(T, int) or T < 1:
            raise ValueError('depths and T must be positive integers')
        if mlp_ratios <= 0 or int(embed_dims * mlp_ratios) < 1:
            raise ValueError('mlp_ratios must produce a positive hidden dimension')
        if backend not in ('torch', 'cupy'):
            raise ValueError('backend must be torch or cupy')
        # CIFAR/baseline define DropPath but never use it. Keep their effective
        # zero stochastic depth, and reject unsupported nonzero dropout settings.
        if any(v not in (None, 0, 0.) for v in
               (drop_rate, attn_drop_rate, drop_path_rate, drop_block_rate)):
            raise ValueError('This adaptation uses zero dropout / drop path, as in the effective CIFAR baseline')
        self.num_classes = num_classes
        self.num_features = embed_dims
        self.depths = depths
        self.T = T
        self.patch_embed = SpikingTokenizer(img_size_h, img_size_w, patch_size,
                                             in_channels, embed_dims, backend)
        feature_size = max(self.patch_embed.H, self.patch_embed.W)
        # num_heads is accepted for baseline config compatibility; there is no QK attention.
        # Do not register unused LayerNorm parameters: they break ordinary DDP backward.
        self.block = nn.ModuleList([
            SpikingTransformer(embed_dims, mlp_ratios, feature_size, backend)
            for _ in range(depths)
        ])
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        x, _ = self.patch_embed(x)
        for block in self.block:
            x = block(x)
        return x.flatten(3).mean(3)

    def forward(self, x):
        if x.ndim != 5 or x.shape[2] != self.patch_embed.C:
            raise ValueError('Expected event frames shaped [B, T, in_channels, H, W]')
        if min(x.shape) < 1:
            raise ValueError('Event frame dimensions must be nonempty')
        # T is dataset metadata. Use the actual sequence length, including T_train
        # subsampling; never repeat a static frame or fold time into the batch.
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        return self.head(self.forward_features(x).mean(0))


@register_model
def spiking_gsap(pretrained=False, cache_dir=None, **kwargs):
    if pretrained:
        raise ValueError('No pretrained UCF101-DVS GSAP weights are provided; load your own checkpoint')
    model = vit_snn(**kwargs)
    model.default_cfg = _cfg(input_size=(model.patch_embed.C, *model.patch_embed.image_size),
                             num_classes=model.num_classes, crop_pct=1.0)
    return model
