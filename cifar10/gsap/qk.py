"""QKFormer with fixed GSAP attention at all three resolutions.

Only attention changes: stage widths, embedding, MLP and classifier follow the
CIFAR-10 QKFormer baseline. GSAP retains the computation in gsap/spiking.py; its axial
kernel coverage is configured separately for each QKFormer stage.
"""
import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
from timm.models import create_model

__all__ = ['qk_gsap']

class GSAPAttention(nn.Module):
    def __init__(self, dim, feature_size=8, lif_backend='cupy'):
        super().__init__()
        assert dim >= 1 and feature_size >= 1
        self.feature_size = feature_size
        kernel_size = 2 * feature_size - 1
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=lif_backend)
        self.feature_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.feature_bn = nn.BatchNorm2d(dim)
        self.feature_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=lif_backend)
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.local_bn = nn.BatchNorm2d(dim)
        self.row_conv = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), padding=(0, feature_size - 1), groups=dim, bias=False)
        self.row_bn = nn.BatchNorm2d(dim)
        self.row_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=lif_backend)
        self.column_conv = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=(feature_size - 1, 0), groups=dim, bias=False)
        self.column_bn = nn.BatchNorm2d(dim)
        self.gate_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.gate_bn = nn.BatchNorm2d(dim)
        self.gate_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=lif_backend)
        self.attn_bn = nn.BatchNorm2d(dim)
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=lif_backend)
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
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., lif_backend='cupy'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)
        self.mlp1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=lif_backend)

        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)
        self.mlp2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=lif_backend)

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


class TokenSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, feature_size=8, lif_backend='cupy'):
        super().__init__()
        self.tssa = GSAPAttention(dim, feature_size=feature_size, lif_backend=lif_backend)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop, lif_backend=lif_backend)

    def forward(self, x):

        x = x + self.tssa(x)
        # print(torch.unique(x))
        x = x + self.mlp(x)
        # print(torch.unique(x))

        return x


class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, feature_size=8, lif_backend='cupy'):
        super().__init__()
        self.ssa = GSAPAttention(dim, feature_size=feature_size, lif_backend=lif_backend)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop, lif_backend=lif_backend)

    def forward(self, x):

        x = x + self.ssa(x)
        x = x + self.mlp(x)

        return x


class PatchEmbedInit(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256, lif_backend='cupy'):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=lif_backend)

        self.proj1_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 1)
        self.proj1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=lif_backend)

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims //1, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=lif_backend)


    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W)
        x = self.proj_lif(x).flatten(0, 1)

        x_feat = x
        x = self.proj1_conv(x)
        x = self.proj1_bn(x).reshape(T, B, -1, H, W)
        x = self.proj1_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H, W).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x


class PatchEmbeddingStage(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256, lif_backend='cupy'):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=lif_backend)

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=lif_backend)

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=lif_backend)

    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res

        x = x.flatten(0, 1).contiguous()
        x_feat = x

        x = self.proj3_conv(x)
        x = self.proj3_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x)
        x = self.proj4_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//2, W//2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x


class spiking_transformer(nn.Module):
    def __init__(self,
                 img_size_h=32, img_size_w=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dims=384, num_heads=8, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=4, sr_ratios=1, T=4, pretrained_cfg=None,
                 pretrained_cfg_overlay=None, drop_block_rate=None, lif_backend='cupy',
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        if not isinstance(embed_dims, int) or embed_dims < 8 or embed_dims % 4:
            raise ValueError('embed_dims must be an integer >= 8 divisible by 4')
        if not isinstance(depths, int) or depths < 3:
            raise ValueError('depths must be >= 3 to include both attention stages')
        if img_size_h < 4 or img_size_w < 4 or img_size_h % 4 or img_size_w % 4:
            raise ValueError('QKFormer image dimensions must be positive multiples of 4')
        if lif_backend not in ('torch', 'cupy'):
            raise ValueError("lif_backend must be 'torch' or 'cupy'")
        if drop_block_rate not in (None, 0.):
            raise ValueError('QKFormer does not implement DropBlock')
        num_heads = [num_heads] * 3 if isinstance(num_heads, int) else num_heads
        # PatchEmbedInit preserves resolution; each later embedding halves it.
        feature_sizes = [max(img_size_h, img_size_w) // scale for scale in (1, 2, 4)]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed1 = PatchEmbedInit(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims // 4, lif_backend=lif_backend)

        # stage1: QKFormers
        stage1 = nn.ModuleList([TokenSpikingTransformer(
            dim=embed_dims // 4, num_heads=num_heads[0], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios, feature_size=feature_sizes[0], lif_backend=lif_backend)
            for j in range(1)])

        patch_embed2 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims // 2, lif_backend=lif_backend)

        stage2 = nn.ModuleList([TokenSpikingTransformer(
            dim=embed_dims // 2, num_heads=num_heads[1], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios, feature_size=feature_sizes[1], lif_backend=lif_backend)
            for j in range(1)])

        patch_embed3 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims, lif_backend=lif_backend)

        stage3 = nn.ModuleList([SpikingTransformer(
            dim=embed_dims, num_heads=num_heads[2], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios, feature_size=feature_sizes[2], lif_backend=lif_backend)
            for j in range(depths - 2)])

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"patch_embed3", patch_embed3)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage2", stage2)
        setattr(self, f"stage3", stage3)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        stage1 = getattr(self, f"stage1")
        patch_embed1 = getattr(self, f"patch_embed1")
        stage2 = getattr(self, f"stage2")
        patch_embed2 = getattr(self, f"patch_embed2")
        stage3 = getattr(self, f"stage3")
        patch_embed3 = getattr(self, f"patch_embed3")

        x = patch_embed1(x)
        for blk in stage1:
            x = blk(x)

        x = patch_embed2(x)
        for blk in stage2:
            x = blk(x)

        x = patch_embed3(x)
        for blk in stage3:
            x = blk(x)

        return x.flatten(3).mean(3)

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))

        return x


@register_model
def qk_gsap(pretrained=False, cache_dir=None, **kwargs):
    """QKFormer with GSAP in every attention block; baseline widths and depths."""
    if pretrained:
        raise ValueError('No pretrained weights are provided for qk_gsap')
    model = spiking_transformer(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


