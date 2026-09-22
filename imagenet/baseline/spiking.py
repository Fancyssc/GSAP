import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial

__all__ = ['Spikingformer']


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

class AdaptiveBinarySimilarity2_1(nn.Module):
    """
    Implements: A = alpha*(QK^T) - beta*(q_sum + k_sum) + gamma
    where alpha, beta > 0 and gamma is signed (can be negative).
    """

    def __init__(self, num_heads, init_alpha=0.5, init_beta_q=0.5, init_beta_k=0.5, init_gamma=0.0):
        super().__init__()
        self.num_heads = num_heads

        self.alpha_logit = nn.Parameter(torch.ones(1, 1, num_heads, 1, 1) * init_alpha)
        self.beta_q_logit = nn.Parameter(torch.ones(1, 1, num_heads, 1, 1) * init_beta_q)
        self.beta_k_logit = nn.Parameter(torch.ones(1, 1, num_heads, 1, 1) * init_beta_k)
        self.gamma = nn.Parameter(torch.ones(1, 1, num_heads, 1, 1) * init_gamma)
        # scale: if None, use 1/sqrt(D_head)
        self.scale = 0.125
    def forward(self, q, k):
        D_head = q.shape[-1]

        qk_dot = q @ k.transpose(-2, -1)                   # (T,B,H,N,N)
        q_sum  = q.sum(dim=-1, keepdim=True)               # (T,B,H,N,1)
        k_sum  = k.sum(dim=-1, keepdim=True).transpose(-2, -1)  # (T,B,H,1,N)

        qk_mean = qk_dot.mean(dim=(0, 1, 3, 4), keepdim=True).detach()  # (1,1,H,1,1)
        q_mean  = q_sum.mean(dim=(0, 1, 3, 4), keepdim=True).detach()
        k_mean  = k_sum.mean(dim=(0, 1, 3, 4), keepdim=True).detach()

        scale_q = qk_mean / q_mean
        scale_k = qk_mean / k_mean
        scale_g = qk_mean


        beta_q = torch.sigmoid(self.beta_q_logit)* scale_q
        beta_k = torch.sigmoid(self.beta_k_logit) * scale_k
        alpha =  torch.sigmoid(self.alpha_logit)
        gamma = self.gamma.tanh()*scale_g
        attn = alpha * qk_dot - beta_q * q_sum  -beta_k * k_sum + gamma
        scale = (D_head ** -0.5) if (self.scale is None) else self.scale
        attn = attn * scale
        return attn
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
        # self.hsa = AdaptiveBinarySimilarity2_1(num_heads=num_heads)


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
        # attn = self.hsa(q,k)
        # x = attn @ v

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0,1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T,B,C,H,W)

        return x

class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpikingSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
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
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)

        self.proj1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj1_conv = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims//4)

        self.proj2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj2_conv = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims//2)

        self.proj3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)

        self.proj4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        T, B, C, H, W = x.shape

        # stem: Conv -> BN
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()

        # stage 1: MP -> LIF -> Conv -> BN
        x = self.maxpool1(x.flatten(0, 1)).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj1_lif(x).flatten(0, 1).contiguous()
        x = self.proj1_conv(x)
        x = self.proj1_bn(x).reshape(T, B, -1, H // 2, W // 2).contiguous()

        # stage 2: MP -> LIF -> Conv -> BN
        x = self.maxpool2(x.flatten(0, 1)).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.proj2_lif(x).flatten(0, 1).contiguous()
        x = self.proj2_conv(x)
        x = self.proj2_bn(x).reshape(T, B, -1, H // 4, W // 4).contiguous()

        # stage 3: MP -> LIF -> Conv -> BN
        x = self.maxpool3(x.flatten(0, 1)).reshape(T, B, -1, H // 8, W // 8).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()
        x = self.proj3_conv(x)
        x = self.proj3_bn(x).reshape(T, B, -1, H // 8, W // 8).contiguous()

        # stage 4: MP -> LIF -> Conv -> BN
        x = self.maxpool4(x.flatten(0, 1)).reshape(T, B, -1, H // 16, W // 16).contiguous()
        x = self.proj4_lif(x).flatten(0, 1).contiguous()
        x = self.proj4_conv(x)
        x = self.proj4_bn(x).reshape(T, B, -1, H // 16, W // 16).contiguous()

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)

    # def forward(self, x):
        # T, B, C, H, W = x.shape
        # x = self.proj_conv(x.flatten(0, 1))
        # x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        #
        # x = self.proj1_lif(x).flatten(0,1).contiguous()
        # x = self.maxpool1(x)
        # x = self.proj1_conv(x)
        # x = self.proj1_bn(x).reshape(T, B, -1, H//2, W//2).contiguous()
        #
        # x = self.proj2_lif(x).flatten(0, 1).contiguous()
        # x = self.maxpool2(x)
        # x = self.proj2_conv(x)
        # x = self.proj2_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
        #
        # x = self.proj3_lif(x).flatten(0, 1).contiguous()
        # x = self.maxpool3(x)
        # x = self.proj3_conv(x)
        # x = self.proj3_bn(x).reshape(T, B, -1, H//8, W//8).contiguous()
        #
        # x = self.proj4_lif(x).flatten(0, 1).contiguous()
        # x = self.maxpool4(x)
        # x = self.proj4_conv(x)
        # x = self.proj4_bn(x).reshape(T, B, -1, H//16, W//16).contiguous()
        #
        # H, W = H // self.patch_size[0], W // self.patch_size[1]
        # return x, (H, W)

class vit_snn(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T = 4, pretrained_cfg= None
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

        block = nn.ModuleList([SpikingTransformer(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

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
        attn = None
        for blk in block:
            x = blk(x)
        return x.flatten(3).mean(3)

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


@register_model
def Spikingformer(pretrained=False, **kwargs):
    model = vit_snn(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

from timm.models import create_model

