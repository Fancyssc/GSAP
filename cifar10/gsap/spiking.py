import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
from timm.models import create_model
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
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
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        return x


class GSAPAttention(nn.Module):
    def __init__(self, dim, feature_size=8):
        super().__init__()
        assert dim >= 1 and feature_size >= 1
        self.feature_size = feature_size
        kernel_size = 2 * feature_size - 1

        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.feature_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.feature_bn = nn.BatchNorm2d(dim)
        self.feature_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.local_bn = nn.BatchNorm2d(dim)

        self.row_conv = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), padding=(0, feature_size - 1), groups=dim, bias=False)
        self.row_bn = nn.BatchNorm2d(dim)
        self.row_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.column_conv = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=(feature_size - 1, 0), groups=dim, bias=False)
        self.column_bn = nn.BatchNorm2d(dim)

        self.gate_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.gate_bn = nn.BatchNorm2d(dim)
        self.gate_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.attn_bn = nn.BatchNorm2d(dim)
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

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
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, feature_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = GSAPAttention(dim, feature_size=feature_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SpikingTokenizer(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.block0_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.block0_bn = nn.BatchNorm2d(embed_dims // 8)

        self.block1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.block1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn = nn.BatchNorm2d(embed_dims // 4)

        self.block2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.block2_conv = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn = nn.BatchNorm2d(embed_dims // 2)

        self.block3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.block3_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block3_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn = nn.BatchNorm2d(embed_dims // 1)

        self.block4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.block4_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.block0_conv(x.flatten(0, 1))
        x = self.block0_bn(x).reshape(T, B, -1, H, W)

        x = self.block1_lif(x).flatten(0, 1)
        x = self.block1_conv(x)
        x = self.block1_bn(x).reshape(T, B, -1, H, W)

        x = self.block2_lif(x).flatten(0, 1)
        x = self.block2_conv(x)
        x = self.block2_bn(x).reshape(T, B, -1, H, W)

        x = self.block3_lif(x).flatten(0, 1)
        x = self.block3_mp(x)
        x = self.block3_conv(x)
        x = self.block3_bn(x).reshape(T, B, -1, int(H / 2), int(W / 2))

        x = self.block4_lif(x).flatten(0, 1)
        x = self.block4_mp(x)
        x = self.block4_conv(x)
        x = self.block4_bn(x).reshape(T, B, -1, int(H / 4), int(W / 4))

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class vit_snn(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=4, pretrained_cfg=None,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SpikingTokenizer(img_size_h=img_size_h,
                          img_size_w=img_size_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dims)
        num_patches = patch_embed.num_patches
        block = nn.ModuleList([SpikingTransformer(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios, feature_size=max(img_size_h, img_size_w) // 4)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

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
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x, (H, W) = patch_embed(x)
        for blk in block:
            x = blk(x)
        return x.flatten(3).mean(3)

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x

@register_model
def spiking_gsap(pretrained=False, **kwargs):
    model = vit_snn(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


