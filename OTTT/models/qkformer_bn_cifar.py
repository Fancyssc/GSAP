import torch
import torch.nn as nn
from modules.neuron import OnlineLIFNode, PPPropLIFNode, WrapedSNNOp, reset_etrace

__all__ = [
    'online_qkformer_cifar',
]


def call_op(op, x, require_wrap):
    if isinstance(op, WrapedSNNOp):
        return op(x, require_wrap=require_wrap)
    return op(x)


def take_spike(x, require_wrap):
    if require_wrap:
        return x[:x.shape[0] // 2]
    return x


class Token_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1,
                 single_step_neuron: callable = None, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.grad_with_rate = kwargs.get('grad_with_rate', False)

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = single_step_neuron(**kwargs)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = single_step_neuron(**kwargs)

        self.attn_lif = single_step_neuron(**dict(kwargs, v_threshold=0.5))

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = single_step_neuron(**kwargs)

        if self.grad_with_rate:
            self.q_conv = WrapedSNNOp(self.q_conv, self.q_lif)
            self.k_conv = WrapedSNNOp(self.k_conv, self.k_lif)
            self.proj_conv = WrapedSNNOp(self.proj_conv, self.proj_lif)
        # attn_lif is driven by the parameter-free matmul, so there is no weight to hook


    def forward(self, x, **kwargs):
        require_wrap = kwargs.get('require_wrap', False)
        B, C, H, W = x.shape

        x = x.flatten(2)
        B, C, N = x.shape

        q_conv_out = call_op(self.q_conv, x, require_wrap)
        q_conv_out = self.q_bn(q_conv_out)
        q_conv_out = self.q_lif(q_conv_out, **kwargs)
        q = q_conv_out.unsqueeze(1).reshape(B, self.num_heads, C // self.num_heads, N)

        k_conv_out = call_op(self.k_conv, x, require_wrap)
        k_conv_out = self.k_bn(k_conv_out)
        k_conv_out = self.k_lif(k_conv_out, **kwargs)
        k = k_conv_out.unsqueeze(1).reshape(B, self.num_heads, C // self.num_heads, N)

        q = torch.sum(q, dim=2, keepdim=True)
        attn = self.attn_lif(take_spike(q, require_wrap), **kwargs)
        x = torch.mul(attn, k)

        x = x.flatten(1, 2)
        x = call_op(self.proj_conv, x, require_wrap)
        x = self.proj_bn(x).reshape(-1, C, H, W)
        x = self.proj_lif(x, **kwargs)

        return x


class Spiking_Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1,
                 single_step_neuron: callable = None, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 0.125 if qk_scale is None else qk_scale
        self.grad_with_rate = kwargs.get('grad_with_rate', False)
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = single_step_neuron(**kwargs)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = single_step_neuron(**kwargs)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = single_step_neuron(**kwargs)
        self.attn_lif = single_step_neuron(**dict(kwargs, v_threshold=0.5))

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = single_step_neuron(**kwargs)

        self.qkv_mp = nn.MaxPool1d(4)

        if self.grad_with_rate:
            self.q_conv = WrapedSNNOp(self.q_conv, self.q_lif)
            self.k_conv = WrapedSNNOp(self.k_conv, self.k_lif)
            self.v_conv = WrapedSNNOp(self.v_conv, self.v_lif)
            self.proj_conv = WrapedSNNOp(self.proj_conv, self.proj_lif)
        # attn_lif is driven by the parameter-free matmul, so there is no weight to hook

    def forward(self, x, **kwargs):
        require_wrap = kwargs.get('require_wrap', False)
        B, C, H, W = x.shape

        x = x.flatten(2)
        B, C, N = x.shape

        q_conv_out = call_op(self.q_conv, x, require_wrap)
        q_conv_out = self.q_bn(q_conv_out).contiguous()
        q_conv_out = self.q_lif(q_conv_out, **kwargs)
        q = q_conv_out.transpose(-1, -2).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3).contiguous()

        k_conv_out = call_op(self.k_conv, x, require_wrap)
        k_conv_out = self.k_bn(k_conv_out).contiguous()
        k_conv_out = self.k_lif(k_conv_out, **kwargs)
        k = k_conv_out.transpose(-1, -2).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3).contiguous()

        v_conv_out = call_op(self.v_conv, x, require_wrap)
        v_conv_out = self.v_bn(v_conv_out).contiguous()
        v_conv_out = self.v_lif(v_conv_out, **kwargs)
        v = v_conv_out.transpose(-1, -2).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3).contiguous()

        x = k.transpose(-2,-1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(2, 3).reshape(B, C, N).contiguous()
        x = self.attn_lif(take_spike(x, require_wrap), **kwargs)
        x = call_op(self.proj_conv, x, require_wrap)
        x = self.proj_bn(x).reshape(-1, C, H, W)
        x = self.proj_lif(x, **kwargs)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.,
                 single_step_neuron: callable = None, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1,bias=False)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)
        self.mlp1_lif = single_step_neuron(**kwargs)

        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1,bias=False)
        self.mlp2_bn = nn.BatchNorm2d(out_features)
        self.mlp2_lif = single_step_neuron(**kwargs)

        self.c_hidden = hidden_features
        self.c_output = out_features

        if kwargs.get('grad_with_rate', False):
            self.mlp1_conv = WrapedSNNOp(self.mlp1_conv, self.mlp1_lif)
            self.mlp2_conv = WrapedSNNOp(self.mlp2_conv, self.mlp2_lif)

    def forward(self, x, **kwargs):
        require_wrap = kwargs.get('require_wrap', False)

        x = call_op(self.mlp1_conv, x, require_wrap)
        x = self.mlp1_bn(x)
        x = self.mlp1_lif(x, **kwargs)

        x = call_op(self.mlp2_conv, x, require_wrap)
        x = self.mlp2_bn(x)
        x = self.mlp2_lif(x, **kwargs)

        return x


class TokenSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1,
                 single_step_neuron: callable = None, **kwargs):
        super().__init__()
        attention_factory = kwargs.pop("attention_factory", Token_QK_Attention)
        self.tssa = attention_factory(
            dim, num_heads=num_heads, qk_scale=qk_scale, single_step_neuron=single_step_neuron, **kwargs)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop,
                       single_step_neuron=single_step_neuron, **kwargs)

    def forward(self, x, **kwargs):

        x = x + self.tssa(x, **kwargs)
        x = x + self.mlp(x, **kwargs)

        return x


class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1,
                 single_step_neuron: callable = None, **kwargs):
        super().__init__()
        attention_factory = kwargs.pop("attention_factory", Spiking_Self_Attention)
        self.ssa = attention_factory(
            dim, num_heads=num_heads, qk_scale=qk_scale, single_step_neuron=single_step_neuron, **kwargs)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop,
                       single_step_neuron=single_step_neuron, **kwargs)

    def forward(self, x, **kwargs):

        x = x + self.ssa(x, **kwargs)
        x = x + self.mlp(x, **kwargs)

        return x


class PatchEmbedInit(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4,
                 in_channels=2, embed_dims=256,
                 single_step_neuron: callable = None, **kwargs):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = [patch_size, patch_size]
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif = single_step_neuron(**kwargs)

        self.proj1_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 1)
        self.proj1_lif = single_step_neuron(**kwargs)

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims //1, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = single_step_neuron(**kwargs)

        if kwargs.get('grad_with_rate', False):
            self.proj1_conv = WrapedSNNOp(self.proj1_conv, self.proj1_lif)
            self.proj_res_conv = WrapedSNNOp(self.proj_res_conv, self.proj_res_lif)
        if isinstance(self.proj_lif, PPPropLIFNode):
            # the first conv consumes the raw image, not spikes, so there is no rate
            # to reroute (OTTT leaves it unwrapped), but it still drives proj_lif, so
            # under pp-prop it is still wrapped to obtain eps_f
            self.proj_conv = WrapedSNNOp(self.proj_conv, self.proj_lif, wrap=False)


    def forward(self, x, **kwargs):
        require_wrap = kwargs.get('require_wrap', False)

        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.proj_lif(x, **kwargs)

        x_feat = x
        x = call_op(self.proj1_conv, x, require_wrap)
        x = self.proj1_bn(x)
        x = self.proj1_lif(x, **kwargs)

        x_feat = call_op(self.proj_res_conv, x_feat, require_wrap)
        x_feat = self.proj_res_bn(x_feat).contiguous()
        x_feat = self.proj_res_lif(x_feat, **kwargs)

        x = x + x_feat # shortcut

        return x


class PatchEmbeddingStage(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4,
                 in_channels=2, embed_dims=256,
                 single_step_neuron: callable = None, **kwargs):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = [patch_size, patch_size]
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_lif = single_step_neuron(**kwargs)

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj4_lif = single_step_neuron(**kwargs)

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = single_step_neuron(**kwargs)

        if kwargs.get('grad_with_rate', False):
            self.proj3_conv = WrapedSNNOp(self.proj3_conv, self.proj3_lif)
            if isinstance(self.proj4_lif, PPPropLIFNode):
                self.proj4_conv = WrapedSNNOp(
                    self.proj4_conv, self.proj4_lif,
                    post_op=nn.Sequential(self.proj4_bn, self.proj4_maxpool))
            else:
                self.proj4_conv = WrapedSNNOp(self.proj4_conv)
            self.proj_res_conv = WrapedSNNOp(self.proj_res_conv, self.proj_res_lif)

    def forward(self, x, **kwargs):
        require_wrap = kwargs.get('require_wrap', False)
        # Downsampling + Res

        x = x.contiguous()
        x_feat = x

        x = call_op(self.proj3_conv, x, require_wrap)
        x = self.proj3_bn(x).contiguous()
        x = self.proj3_lif(x, **kwargs)

        x = call_op(self.proj4_conv, x, require_wrap)
        if not getattr(self.proj4_conv, 'includes_post_op', False):
            x = self.proj4_bn(x)
            x = self.proj4_maxpool(x)
        x = x.contiguous()
        x = self.proj4_lif(x, **kwargs)

        x_feat = call_op(self.proj_res_conv, x_feat, require_wrap)
        x_feat = self.proj_res_bn(x_feat).contiguous()
        x_feat = self.proj_res_lif(x_feat, **kwargs)

        x = x + x_feat # shortcut

        return x


class spiking_transformer(nn.Module):
    def __init__(self,
                 img_size_h=32, img_size_w=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dims=384, num_heads=8, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=4, sr_ratios=1, pretrained_cfg=None,
                 single_step_neuron: callable = None, **kwargs
                 ):
        super().__init__()
        single_step_neuron = single_step_neuron or OnlineLIFNode
        if not isinstance(depths, int) or depths < 3:
            raise ValueError("depths must be >= 3")
        if not isinstance(embed_dims, int) or embed_dims < 8 or embed_dims % 8:
            raise ValueError("embed_dims must be >= 8 and divisible by 8")
        self.num_classes = num_classes
        self.depths = depths
        self.single_step_neuron = single_step_neuron
        num_heads = [num_heads] * 3 if isinstance(num_heads, int) else num_heads
        if len(num_heads) != 3 or any(h < 1 for h in num_heads):
            raise ValueError("num_heads must contain three positive counts")
        feature_sizes = [(max(img_size_h, img_size_w) + scale - 1) // scale
                         for scale in (1, 2, 4)]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed1 = PatchEmbedInit(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims // 4,
                                       single_step_neuron=single_step_neuron, **kwargs)

        stage1 = nn.ModuleList([TokenSpikingTransformer(
            dim=embed_dims // 4, num_heads=num_heads[0], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios, feature_size=feature_sizes[0],
            single_step_neuron=single_step_neuron, **kwargs)
            for j in range(1)])

        patch_embed2 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims // 2,
                                       single_step_neuron=single_step_neuron, **kwargs)

        stage2 = nn.ModuleList([TokenSpikingTransformer(
            dim=embed_dims // 2, num_heads=num_heads[1], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios, feature_size=feature_sizes[1],
            single_step_neuron=single_step_neuron, **kwargs)
            for j in range(1)])

        patch_embed3 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims,
                                       single_step_neuron=single_step_neuron, **kwargs)

        stage3 = nn.ModuleList([SpikingTransformer(
            dim=embed_dims, num_heads=num_heads[2], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios, feature_size=feature_sizes[2],
            single_step_neuron=single_step_neuron, **kwargs)
            for j in range(depths - 2)])

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"patch_embed3", patch_embed3)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage2", stage2)
        setattr(self, f"stage3", stage3)

        # classification head
        self.grad_with_rate = kwargs.get('grad_with_rate', False)
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        if self.grad_with_rate and num_classes > 0:
            self.head = WrapedSNNOp(self.head)  # readout: drives no neuron

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in' if m.groups == m.in_channels else 'fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, **kwargs):
        stage1 = getattr(self, f"stage1")
        patch_embed1 = getattr(self, f"patch_embed1")
        stage2 = getattr(self, f"stage2")
        patch_embed2 = getattr(self, f"patch_embed2")
        stage3 = getattr(self, f"stage3")
        patch_embed3 = getattr(self, f"patch_embed3")

        x = patch_embed1(x, **kwargs)
        for blk in stage1:
            x = blk(x, **kwargs)

        x = patch_embed2(x, **kwargs)
        for blk in stage2:
            x = blk(x, **kwargs)

        x = patch_embed3(x, **kwargs)
        for blk in stage3:
            x = blk(x, **kwargs)

        return x.flatten(2).mean(2)

    def forward(self, x, **kwargs):
        if x.ndim != 4:
            raise ValueError('spiking_transformer expects [B, C, H, W]')
        require_wrap = self.grad_with_rate and self.training
        kwargs['require_wrap'] = require_wrap
        kwargs['output_type'] = 'spike_rate' if require_wrap else 'spike'

        if kwargs.get('init', False):  # weights run before the neurons, so clear eps_f at the start of the sequence here
            reset_etrace(self)
        x = self.forward_features(x, **kwargs)
        if isinstance(self.head, nn.Identity):
            return take_spike(x, require_wrap)
        return call_op(self.head, x, require_wrap)

    def get_spike(self):
        spikes = []
        for module in self.modules():
            if isinstance(module, self.single_step_neuron) and hasattr(module, "spike"):
                spike = module.spike.cpu()
                spikes.append(spike.reshape(spike.shape[0], -1))
        return spikes


def _qkformer(embed_dims, depths, single_step_neuron: callable = None, **kwargs):
    num_classes = kwargs.pop('num_classes', 10)
    in_channels = kwargs.pop('in_channels', 3)
    in_channels = kwargs.pop('c_in', in_channels)
    return spiking_transformer(
        in_channels=in_channels, num_classes=num_classes,
        embed_dims=embed_dims, depths=depths,
        single_step_neuron=single_step_neuron, **kwargs)


def online_qkformer_cifar(pretrained=False, progress=True,
                          single_step_neuron: callable = None, **kwargs):
    return _qkformer(
        embed_dims=kwargs.pop('embed_dims', 384), depths=kwargs.pop('depths', 4),
        single_step_neuron=single_step_neuron, **kwargs)
