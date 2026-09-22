import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

__all__ = ['MetaSpikingformer']


class BNAndPadLayer(nn.Module):
    def __init__(
        self,
        pad_pixels,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, x):
        x = self.bn(x)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            x = F.pad(x, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            x[:, :, 0:self.pad_pixels, :] = pad_values
            x[:, :, -self.pad_pixels:, :] = pad_values
            x[:, :, :, 0:self.pad_pixels] = pad_values
            x[:, :, :, -self.pad_pixels:] = pad_values
        return x


class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        conv1x1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channels)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 0, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=1, bias=bias),
            nn.BatchNorm2d(out_channels),
        )
        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepConv(nn.Module):
    def __init__(
        self,
        dim,
        expansion_ratio=2,
        bias=False,
        kernel_size=7,
        padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)

        self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)

        self.lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )
        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.lif1(x)
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)

        x = self.lif2(x)
        x = self.dwconv(x.flatten(0, 1))
        x = self.bn2(self.pwconv2(x)).reshape(T, B, -1, H, W)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.conv = SepConv(dim=dim)

        self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.conv1 = nn.Conv2d(dim, mlp_hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mlp_hidden_dim)

        self.lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.conv2 = nn.Conv2d(mlp_hidden_dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.conv(x) + x
        x_feat = x

        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)

        x = x_feat + x
        return x


class MS_MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)

        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)

        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        x = x.flatten(3)
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.hidden_features, N).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, self.out_features, H, W).contiguous()

        return x


class MS_Attention_RepConv_qkv_id(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        sr_ratio=1,
    ):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125 if qk_scale is None else qk_scale

        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.attn_lif = MultiStepLIFNode(
            tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy'
        )
        self.proj_conv = nn.Sequential(
            RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        x = self.head_lif(x)

        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        q = self.q_lif(q).flatten(3)
        q = q.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k = self.k_lif(k).flatten(3)
        k = k.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v = self.v_lif(v).flatten(3)
        v = v.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, H, W)
        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        return x


class MetaFormerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()

        self.attn = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class MS_DownSampling(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.encode_bn = nn.BatchNorm2d(embed_dims)

        if not first_layer:
            self.encode_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, _, _, _ = x.shape

        if hasattr(self, 'encode_lif'):
            x = self.encode_lif(x)

        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()
        return x


class vit_snn_metaformer(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dims=256, # 96 192 384 384
        num_heads=8,
        mlp_ratios=2,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        depths=4,
        sr_ratios=1,
        T=4,
        pretrained_cfg=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        if isinstance(embed_dims, int):
            stage_dims = [embed_dims // 4, embed_dims // 4, embed_dims//2, embed_dims]
        else:
            stage_dims = list(embed_dims)
            assert len(stage_dims) == 4, 'embed_dims should be an int or a list of length 4.'

        if isinstance(mlp_ratios, (int, float)):
            stage_mlp_ratio = mlp_ratios
        else:
            stage_mlp_ratio = mlp_ratios[-1]

        total_depth = int(depths)
        stage4_depth = max(total_depth // 4, 1)
        stage3_depth = max(total_depth - stage4_depth, 1)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, stage3_depth + stage4_depth)]

        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=stage_dims[0],
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
        )
        self.ConvBlock1_1 = nn.ModuleList([MS_ConvBlock(dim=stage_dims[0], mlp_ratio=stage_mlp_ratio)])

        self.downsample1_2 = MS_DownSampling(
            in_channels=stage_dims[0],
            embed_dims=stage_dims[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )
        self.ConvBlock1_2 = nn.ModuleList([MS_ConvBlock(dim=stage_dims[1], mlp_ratio=stage_mlp_ratio)])

        self.downsample2 = MS_DownSampling(
            in_channels=stage_dims[1],
            embed_dims=stage_dims[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )
        self.ConvBlock2_1 = nn.ModuleList([MS_ConvBlock(dim=stage_dims[2], mlp_ratio=stage_mlp_ratio)])
        self.ConvBlock2_2 = nn.ModuleList([MS_ConvBlock(dim=stage_dims[2], mlp_ratio=stage_mlp_ratio)])

        self.downsample3 = MS_DownSampling(
            in_channels=stage_dims[2],
            embed_dims=stage_dims[2],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
        )
        self.block3 = nn.ModuleList([
            MetaFormerBlock(
                dim=stage_dims[2],
                num_heads=num_heads,
                mlp_ratio=stage_mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[j],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios,
            )
            for j in range(2)
        ])

        self.downsample4 = MS_DownSampling(
            in_channels=stage_dims[2],
            embed_dims=stage_dims[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
        )
        self.block4 = nn.ModuleList([
            MetaFormerBlock(
                dim=stage_dims[3],
                num_heads=num_heads,
                mlp_ratio=stage_mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[2 + j],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios,
            )
            for j in range(2)
        ])

        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.head = nn.Linear(stage_dims[3], num_classes) if num_classes > 0 else nn.Identity()

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
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)

        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)

        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)

        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)

        return x.flatten(3).mean(3)

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(self.lif(x).mean(0))
        return x


@register_model
def MetaSpikingformer(pretrained=False, **kwargs):
    model = vit_snn_metaformer(**kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def metaspikingformer(pretrained=False, **kwargs):
    model = vit_snn_metaformer(**kwargs)
    model.default_cfg = _cfg()
    return model


