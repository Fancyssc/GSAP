"""N-Caltech101 QKFormer: two-stage DVS backbone with GSAP attention.

Preserves baseline embeddings, MLPs, widths (128/256) and one block per stage.
Consumes real event frames [B, T, 2, H, W]; reset neurons between batches.
"""
import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

__all__ = ['qk_gsap']


class GSAPAttention(nn.Module):
    def __init__(self, dim, feature_size=8, backend='cupy'):
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


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., backend='cupy'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)
        self.mlp1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)
        self.mlp2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)
        x = self.mlp1_lif(x)

        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        x = self.mlp2_lif(x)
        return x


class PatchEmbedInit(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256, backend='cupy'):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.proj1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 4)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.proj2_conv = nn.Conv2d(embed_dims//4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims // 2)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.proj3_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.proj_res_conv = nn.Conv2d(embed_dims // 4, embed_dims, kernel_size=1, stride=4, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)


    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res
        # x_feat = x.flatten(0, 1)
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W)
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = self.maxpool1(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj1_lif(x).flatten(0, 1).contiguous()

        x_feat = x
        x = self.proj2_conv(x)
        x = self.proj2_bn(x)
        x = self.maxpool2(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.proj2_lif(x).flatten(0, 1).contiguous()

        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.maxpool3(x).reshape(T, B, -1, H // 8, W // 8).contiguous()
        x = self.proj3_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//8, W//8).contiguous()
        x_feat = self.proj_res_lif(x_feat)
        x = x + x_feat # shortcut

        return x


class PatchEmbeddingStage(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256, backend='cupy'):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res

        x = x.flatten(0, 1).contiguous()
        x_feat = x

        x = self.proj_conv(x)
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x)
        x = self.proj4_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//2, W//2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x


class TokenSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, feature_size=8, backend='cupy'):
        super().__init__()
        self.tssa = GSAPAttention(dim, feature_size=feature_size, backend=backend)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop, backend=backend)

    def forward(self, x):

        x = x + self.tssa(x)
        # print(torch.unique(x))
        x = x + self.mlp(x)
        # print(torch.unique(x))

        return x


class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, feature_size=8, backend='cupy'):
        super().__init__()
        self.ssa = GSAPAttention(dim, feature_size=feature_size, backend=backend)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop, backend=backend)

    def forward(self, x):

        x = x + self.ssa(x)
        x = x + self.mlp(x)

        return x


class vit_snn(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=16,
                 in_channels=2, num_classes=101, embed_dims=256, num_heads=16,
                 mlp_ratios=1., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=2, sr_ratios=1, T=16, pretrained_cfg=None,
                 pretrained_cfg_overlay=None, drop_block_rate=None, backend='cupy'):
        super().__init__()
        if depths != 2:
            raise ValueError('DVS QKFormer uses two stages with one block each; depths must be 2')
        if not isinstance(T, int) or T < 1:
            raise ValueError('T must be a positive integer')
        if embed_dims < 16 or embed_dims % 16:
            raise ValueError('embed_dims must be a positive multiple of 16')
        if min(img_size_h, img_size_w) < 16 or img_size_h % 16 or img_size_w % 16:
            raise ValueError('Image dimensions must be positive multiples of 16')
        if to_2tuple(patch_size) != (16, 16):
            raise ValueError('DVS QKFormer has total stride 16; patch_size must be 16')
        if mlp_ratios <= 0 or int(embed_dims // 2 * mlp_ratios) < 1:
            raise ValueError('mlp_ratios must produce a positive hidden dimension')
        if backend not in ('torch', 'cupy'):
            raise ValueError('backend must be torch or cupy')
        if any(v not in (None, 0, 0.) for v in
               (drop_rate, attn_drop_rate, drop_path_rate, drop_block_rate)):
            raise ValueError('This adaptation uses zero dropout / drop path')
        self.num_classes = num_classes
        self.num_features = embed_dims
        self.depths = depths
        self.T = T
        self.in_channels = in_channels
        self.image_size = (img_size_h, img_size_w)
        self.feature_sizes = (max(self.image_size) // 8, max(self.image_size) // 16)
        self.patch_embed1 = PatchEmbedInit(img_size_h, img_size_w, patch_size,
                                          in_channels, embed_dims // 2, backend)
        self.patch_embed2 = PatchEmbeddingStage(img_size_h, img_size_w, patch_size,
                                               in_channels, embed_dims, backend)
        self.stage1 = nn.ModuleList([TokenSpikingTransformer(
            embed_dims // 2, num_heads, mlp_ratio=mlp_ratios,
            feature_size=self.feature_sizes[0], backend=backend)])
        self.stage2 = nn.ModuleList([SpikingTransformer(
            embed_dims, num_heads, mlp_ratio=mlp_ratios,
            feature_size=self.feature_sizes[1], backend=backend)])
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        x = self.patch_embed1(x)
        for block in self.stage1:
            x = block(x)
        x = self.patch_embed2(x)
        for block in self.stage2:
            x = block(x)
        return x.flatten(3).mean(3)

    def forward(self, x):
        if x.ndim != 5 or x.shape[2] != self.in_channels or min(x.shape) < 1:
            raise ValueError('Expected nonempty event frames [B, T, in_channels, H, W]')
        if tuple(x.shape[-2:]) != self.image_size:
            raise ValueError(f'Expected spatial size {self.image_size}, got {tuple(x.shape[-2:])}')
        # Honor actual sequence length, including T_train subsampling.
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        return self.head(self.forward_features(x).mean(0))


@register_model
def qk_gsap(pretrained=False, cache_dir=None, **kwargs):
    if pretrained:
        raise ValueError('No pretrained weights are provided for qk_gsap')
    model = vit_snn(**kwargs)
    model.default_cfg = _cfg(input_size=(model.in_channels, *model.image_size),
                             num_classes=model.num_classes, crop_pct=1.0)
    return model
