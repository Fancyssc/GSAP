import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
from timm.models import create_model

class SPS(nn.Module):
    def __init__(self, step=4, img_h=32, img_w=32, patch_size=4, in_channels=3, embed_dims=384,backend='cupy', **kwargs):
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size
        self.patch_nums = self.img_h // self.patch_size * self.img_w // self.patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.step = step

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 =  MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif =  MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)


    def forward(self, x):

        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x)  # TB C H W

        x = self.proj_conv1(x.flatten(0, 1))
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x)

        x = self.proj_conv2(x.flatten(0, 1))
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif2(x)
        x = self.maxpool2(x.flatten(0, 1))

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        x = self.maxpool3(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x_feat = x # T B -1, H // 4, W // 4
        x = self.proj_lif3(x)

        x = self.rpe_conv(x.flatten(0, 1))
        x = self.rpe_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = x + x_feat

        return x # T B -1, H // 4, W // 4

# SDSA
class SDSA(nn.Module):
    def __init__(self,embed_dim, step=4,num_heads=12, backend='cupy'):
        super().__init__()
        self.num_heads = num_heads

        self.q_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(embed_dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.k_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(embed_dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.v_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(embed_dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        #special v_thres
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=backend)

        self.talking_heads = nn.Conv1d(num_heads, num_heads, kernel_size=1, stride=1, bias=False)
        # SDSA uses only talking_heads_lif below. Keep the unused convolution
        # for checkpoint compatibility, but exclude its weight from DDP reduction.
        self.talking_heads.requires_grad_(False)
        self.talking_heads_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.proj_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(embed_dim)

        self.shortcut_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

    def forward(self, x):
        # T B -1, H // 4, W // 4

        T, B, C, H, W = x.shape #TB dim H//4 W//4
        N = H * W

        identity = x

        #shortcut
        x = self.shortcut_lif(x)

        x_for_qkv = x.flatten(0, 1)#TB dim H//4 W//4
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_lif(q_conv_out)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_lif(k_conv_out)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out)

        q = (
             q_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous())
        k = ( k_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous())
        v = (v_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous())

        # attn
        kv = k.mul(v)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv) #TB H N C//H
        x = q.mul(kv)

        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        x = (
            self.proj_bn(self.proj_conv(x.flatten(0, 1)))
            .reshape(T, B, C, H, W)
            .contiguous()
        )

        x = x + identity
        return x

class MLP(nn.Module):
    def __init__(self, in_features, step=4,  mlp_ratio = 4.0, out_features=None,mlp_drop=0., backend='cupy'):
        super().__init__()

        self.in_features = in_features
        self.mlp_ratio = mlp_ratio
        self.out_features = out_features or in_features
        self.hidden_features = int(self.in_features * self.mlp_ratio)

        self.fc_conv1 = nn.Conv2d(in_features, self.hidden_features, kernel_size=1, stride=1)
        self.fc_bn1 = nn.BatchNorm2d(self.hidden_features)
        self.fc_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.fc_conv2 = nn.Conv2d(self.hidden_features, self.out_features, kernel_size=1, stride=1)
        self.fc_bn2 = nn.BatchNorm2d(self.out_features)
        self.fc_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

    def forward(self, x):
        # T B -1, H // 4, W // 4
        T, B, C, H, W = x.shape
        identity = x

        x = self.fc_lif1(x)
        x = self.fc_conv1(x.flatten(0, 1))
        x = self.fc_bn1(x).reshape(T, B, -1, H, W)

        x = self.fc_lif2(x)
        x = self.fc_conv2(x.flatten(0, 1))
        x = self.fc_bn2(x).reshape(T, B, -1, H, W)

        return x+identity

# Spikformer block
class SDT_Block_s(nn.Module):
    def __init__(self, embed_dim=384, num_heads=12, step=4, mlp_ratio=4.,mlp_drop=0., backend='cupy'):
        super().__init__()

        self.attn = SDSA(
                embed_dim, step=step, num_heads=num_heads,  backend=backend)
        self.mlp = MLP(step=step,in_features=embed_dim,mlp_ratio=mlp_ratio,out_features=embed_dim,mlp_drop=mlp_drop, backend=backend)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x


class SDTV1(nn.Module):
    def __init__(self, step=4,img_size=32, patch_size=4, in_channels=3, num_classes=10,embed_dim=384,
                 num_heads=12, mlp_ratio=4,mlp_drop=0., depths=4, backend='cupy'):
        super().__init__()
        self.step = step  # time step
        self.num_classes = num_classes
        self.depths = depths
        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)


        patch_embed = SPS(img_h=img_size,
                          img_w=img_size,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dim, backend=backend)

        block = nn.ModuleList([SDT_Block_s(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           mlp_ratio=mlp_ratio,
                                           mlp_drop=mlp_drop,  backend=backend)


                               for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)
        # classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        x = patch_embed(x)
        for blk in block:
            x = blk(x) #TB C H W
        # dim adjustment
        T, B, C, H, W = x.shape
        x = x.flatten(0, 1).flatten(-2, -1) # TB C N

        return x.mean(2).reshape(T, B, C).contiguous() # T B C

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.step, 1, 1, 1, 1)
        x = self.forward_features(x) #T B C
        x = self.head_lif(x)
        x = self.head(x).mean(0) # TET = False
        return x

#### models for static datasets
@register_model
def sdt_baseline(pretrained=False,**kwargs):
    if pretrained:
        raise ValueError('No pretrained Tiny ImageNet weights are provided')
    model = SDTV1(
        step=kwargs.get('step', kwargs.get('T', 4)),
        img_size=kwargs.get('img_size', kwargs.get('img_size_h', 64)),
        patch_size=kwargs.get('patch_size', 4),
        in_channels=kwargs.get('in_channels', 3),
        num_classes=kwargs.get('num_classes', 200),
        embed_dim=kwargs.get('embed_dim', kwargs.get('embed_dims', 384)),
        num_heads=kwargs.get('num_heads', 12),
        mlp_ratio=kwargs.get('mlp_ratio', kwargs.get('mlp_ratios', 4)),
        mlp_drop=kwargs.get('mlp_drop', kwargs.get('drop_rate', 0.0)),
        depths=kwargs.get('depths', 4),
        backend=kwargs.get('backend', 'cupy'),
    )
    model.default_cfg = _cfg(input_size=(3, 64, 64), num_classes=model.num_classes, crop_pct=1.0)
    return model
