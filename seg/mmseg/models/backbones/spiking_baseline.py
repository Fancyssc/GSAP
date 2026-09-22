import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial
import warnings
from collections import OrderedDict

from mmengine.model import BaseModule
from mmengine.logging import print_log
from mmengine.runner import CheckpointLoader
from mmseg.registry import MODELS

__all__ = ['Spikingformer', 'spiking_baseline']


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
        T,B,C,H,W = x.shape

        x = self.mlp1_lif(x)
        x = self.mlp1_conv(x.flatten(0,1))
        x = self.mlp1_bn(x).reshape(T,B,self.c_hidden,H,W).contiguous()

        x = self.mlp2_lif(x)
        x = self.mlp2_conv(x.flatten(0,1))
        x = self.mlp2_bn(x).reshape(T,B,C,H,W).contiguous()

        return x

class SpikingSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)


    def forward(self, x):
        T,B,C,H,W = x.shape

        x = self.proj_lif(x)

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2,-1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0,1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T,B,C,H,W)

        return x

class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.Identity()
        self.attn = SpikingSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


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
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
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
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0: self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0: self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class RepConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            bias=False,
    ):
        super().__init__()
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
            self,
            dim,
            expansion_ratio=2,
            act2_layer=nn.Identity,
            bias=False,
            kernel_size=7,
            padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
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
    def __init__(
            self,
            dim,
            mlp_ratio=4.0,
    ):
        super().__init__()

        self.Conv = SepConv(dim=dim)
        # self.Conv = MHMC(dim=dim)

        self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        # self.conv1 = RepConv(dim, dim*mlp_ratio)
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)
        self.lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        # self.conv2 = RepConv(dim*mlp_ratio, dim)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, 4 * C, H, W)
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


class MS_DownSampling(nn.Module):
    """Conv downsampling block, identical in structure to sdtv2's MS_DownSampling
    (spike -> conv -> bn). Operates on (T, B, C, H, W) SNN tensors."""

    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2,
                 padding=1, first_layer=True):
        super().__init__()
        self.encode_conv = nn.Conv2d(
            in_channels, embed_dims, kernel_size=kernel_size, stride=stride,
            padding=padding)
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, _, _, _ = x.shape
        if hasattr(self, "encode_lif"):
            x = self.encode_lif(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()
        return x


class vit_snn(BaseModule):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T = 4, pretrained_cfg= None,
                 decode_mode='snn', stage_channels=None, init_cfg=None, pretrained=None
                 ):
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None and pretrained is not False:
            raise TypeError('pretrained must be a str or None')

        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        self.decode_mode = decode_mode
        # ===== sdtv2-style stage layout =====
        # stage_channels = [x1, x2, x3, x4] output channels.
        # embed_dim = [s2, s3, s3*2, s4] feeds the 5 downsample stages.
        # Early stages are convolutional (MS_ConvBlock); only the 8 deep
        # transformer blocks (block3: 6, block4: 2) are SpikingTransformer.
        self.stage_channels = stage_channels or [32, 64, 128, 360]
        sc = self.stage_channels
        embed_dim = [sc[1], sc[2], sc[2] * 2, sc[3]]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels, embed_dims=embed_dim[0] // 2,
            kernel_size=7, stride=2, padding=3, first_layer=True)
        self.ConvBlock1_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)])

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2, embed_dims=embed_dim[0],
            kernel_size=3, stride=2, padding=1, first_layer=False)
        self.ConvBlock1_2 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios)])

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0], embed_dims=embed_dim[1],
            kernel_size=3, stride=2, padding=1, first_layer=False)
        self.ConvBlock2_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)])
        self.ConvBlock2_2 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)])

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1], embed_dims=embed_dim[2],
            kernel_size=3, stride=2, padding=1, first_layer=False)
        self.block3 = nn.ModuleList([SpikingTransformer(
            dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(6)])

        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2], embed_dims=embed_dim[3],
            kernel_size=3, stride=1, padding=1, first_layer=False)
        self.block4 = nn.ModuleList([SpikingTransformer(
            dim=embed_dim[3], num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(6, 8)])

        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.head = nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        if self.init_cfg is None:
            print_log(f'No pre-trained weights for '
                      f'{self.__class__.__name__}, '
                      f'training start from scratch')
            self.apply(self._init_weights)
            print_log("Time step: {:}".format(self.T))
            return

        assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                              f'specify `Pretrained` in ' \
                                              f'`init_cfg` in ' \
                                              f'{self.__class__.__name__} '
        ckpt = CheckpointLoader.load_checkpoint(
            self.init_cfg['checkpoint'], logger=None, map_location='cpu')
        if 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            _state_dict = ckpt['model']
        else:
            _state_dict = ckpt

        model_state = self.state_dict()
        state_dict = OrderedDict()
        for k, v in _state_dict.items():
            if k.startswith('backbone.'):
                k = k[9:]
            if k in model_state and model_state[k].shape == v.shape:
                state_dict[k] = v
        self.load_state_dict(state_dict, strict=False)
        print_log("--------------Successfully load checkpoint for BACKBONE------------")
        print_log("Loaded backbone tensors: {:}".format(len(state_dict)))
        print_log("Time step: {:}".format(self.T))

    def forward_features(self, x):
        # ===== sdtv2-style forward: x1/x2 from downsampling stages,
        # x3 after the 2 light blocks of stage2, x4 after 6+2 heavy blocks =====
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x1 = x

        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)
        x2 = x

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)
        x3 = x

        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)

        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)
        x4 = x

        features = [x1, x2, x3, x4]
        if self.decode_mode == 'snn':
            return [feat.mean(0) for feat in features]
        return [feat.flatten(0, 1) for feat in features]

    def forward(self, x):
        T = self.T
        x = (x.unsqueeze(0)).repeat(T, 1, 1, 1, 1)
        return self.forward_features(x)


@register_model
def Spikingformer(pretrained=False, **kwargs):
    model = vit_snn(
        pretrained=pretrained,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@MODELS.register_module()
def spiking_baseline(pretrained=None, **kwargs):
    img_size_h = kwargs.pop("img_size_h", 224)
    img_size_w = kwargs.pop("img_size_w", 224)
    if "embed_dims" in kwargs:
        embed_dims = kwargs.pop("embed_dims")
    else:
        embed_dims = kwargs.pop("embed_dim", 512)

    model = vit_snn(
        img_size_h=img_size_h,
        img_size_w=img_size_w,
        patch_size=kwargs.pop("patch_size", 16),
        embed_dims=embed_dims,
        stage_channels=kwargs.pop("stage_channels", None),
        num_heads=kwargs.pop("num_heads", 8),
        mlp_ratios=kwargs.pop("mlp_ratios", 4),
        qkv_bias=kwargs.pop("qkv_bias", False),
        depths=kwargs.pop("depths", 8),
        sr_ratios=kwargs.pop("sr_ratios", 1),
        pretrained=pretrained,
        **kwargs,
    )
    return model

from timm.models import create_model

if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224).cuda()
    model = create_model(
        'Spikingformer',
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=512, num_heads=8, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=8, sr_ratios=1,
        T = 4
    ).cuda()

    model.eval()
    y = model(x)
    print(y.shape)
    print('Test Good!')
