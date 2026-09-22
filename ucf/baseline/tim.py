from einops import rearrange
import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, LIFNode
from timm.models import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _cfg


class SPS(nn.Module):
    def __init__(self, step=4, img_h=128, img_w=128, patch_size=4, in_channels=2, embed_dims=384, **kwargs):
        super().__init__()
        self.step = step
        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size
        self.patch_nums = self.img_h // self.patch_size * self.img_w // self.patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x)

        x = self.maxpool(x.flatten(0, 1))
        H, W = H // 2, W // 2
        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x)

        x = self.maxpool1(x.flatten(0, 1))
        H, W = H // 2, W // 2
        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif2(x)

        x = self.maxpool2(x.flatten(0, 1))
        H, W = H // 2, W // 2
        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif3(x)

        x = self.maxpool3(x.flatten(0, 1))
        H, W = H // 2, W // 2
        x = x.reshape(T, B, -1, H, W).contiguous()
        x_feat = x

        x = self.rpe_conv(x.flatten(0, 1))
        x = self.rpe_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.rpe_lif(x)

        x = x + x_feat
        x = x.flatten(-2)

        return x


class TIMv1(nn.Module):
    def __init__(self, step=4, in_channels=16, TIM_alpha=0.5):
        super().__init__()
        self.T = step
        self.interactor = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )
        self.in_lif = LIFNode(tau=2.0, v_threshold=0.3, detach_reset=True)
        self.out_lif = LIFNode(tau=2.0, v_threshold=0.5, detach_reset=True)
        self.tim_alpha = TIM_alpha

    def forward(self, x):
        T, B, H, N, CoH = x.shape

        output = []
        x_tim = torch.empty_like(x[0])
        for i in range(T):
            if i == 0:
                x_tim = x[i]
                output.append(x_tim)
            else:
                x_tim = x_tim.flatten(0, 1)   # [BH, CoH, N]
                x_tim = x_tim.permute(0, 2, 1).contiguous()   # [BH, N, CoH]
                x_tim = self.interactor(x_tim).transpose(-2, -1).reshape(B, H, N, CoH).contiguous()
                x_tim = self.in_lif(x_tim) * self.tim_alpha + x[i] * (1 - self.tim_alpha)
                x_tim = self.out_lif(x_tim)
                output.append(x_tim)

        return torch.stack(output)


class SSA(nn.Module):
    def __init__(self, embed_dim, step=4, num_heads=12, attn_scale=0.125, num_tokens=16):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} should be divided by num_heads {num_heads}."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = attn_scale
        self.step = step

        self.q_linear = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.TIMv1 = TIMv1(step=self.step)

        self.k_linear = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_linear = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(embed_dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_linear = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(embed_dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_linear_out = self.q_linear(x_for_qkv)
        q_linear_out = self.q_bn(q_linear_out).reshape(T, B, C, N).contiguous()
        q_linear_out = self.q_lif(q_linear_out).transpose(-2, -1)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 1, 3, 2, 4).contiguous()

        q = self.TIMv1(q)

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out).reshape(T, B, C, N).contiguous()
        k_linear_out = self.k_lif(k_linear_out).transpose(-2, -1)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads)
        k = k.permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out).reshape(T, B, C, N).contiguous()
        v_linear_out = self.v_lif(v_linear_out).transpose(-2, -1)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads)
        v = v.permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(-2, -1).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = self.proj_bn(self.proj_linear(x.flatten(0, 1))).reshape(T, B, C, N).contiguous()
        x = self.proj_lif(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, step=4, mlp_ratio=4.0, out_features=None, mlp_drop=0.):
        super().__init__()
        self.out_features = out_features or in_features
        self.hidden_features = int(in_features * mlp_ratio)
        self.mlp_drop = mlp_drop
        self.step = step

        self.id = nn.Identity()

        self.fc1_linear = nn.Conv1d(in_features, self.hidden_features, kernel_size=1, stride=1, bias=False)
        self.fc1_bn = nn.BatchNorm1d(self.hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.fc2_linear = nn.Conv1d(self.hidden_features, self.out_features, kernel_size=1, stride=1, bias=False)
        self.fc2_bn = nn.BatchNorm1d(self.out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, N = x.shape
        x = x.flatten(0, 1)

        x = self.fc1_linear(x)
        x = self.fc1_bn(x).reshape(T, B, self.hidden_features, N).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_linear(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, N).contiguous()
        x = self.fc2_lif(x)
        return x


class Block(nn.Module):
    def __init__(self, embed_dim=384, num_heads=12, step=4, mlp_ratio=4., attn_scale=0.125, mlp_drop=0., num_tokens=16):
        super().__init__()
        self.attn = SSA(embed_dim, step=step, num_heads=num_heads, attn_scale=attn_scale, num_tokens=num_tokens)
        self.mlp = MLP(step=step, in_features=embed_dim, mlp_ratio=mlp_ratio, out_features=embed_dim, mlp_drop=mlp_drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Spikformer(nn.Module):
    def __init__(
        self,
        step=4,
        img_size=128,
        patch_size=4,
        in_channels=2,
        num_classes=101,
        attn_scale=0.125,
        embed_dim=384,
        num_heads=16,
        mlp_ratio=4,
        attn_drop=0.,
        depths=4,
        **kwargs,
    ):
        super().__init__()
        self.T = step
        self.num_classes = num_classes
        self.depths = depths

        patch_embed = SPS(
            step=step,
            embed_dims=embed_dim,
            img_h=img_size,
            img_w=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
        )
        num_tokens = (img_size // 16) * (img_size // 16)
        block = nn.ModuleList([
            Block(
                step=step,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_scale=attn_scale,
                num_tokens=num_tokens,
            )
            for _ in range(depths)
        ])

        self.patch_embed = patch_embed
        self.block = block

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        for blk in self.block:
            x = blk(x)
        return x.mean(-1)

    def forward(self, x):
        x = x.transpose(0, 1).contiguous()
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


@register_model
def TIM(pretrained=False, **kwargs):
    model = Spikformer(
        step=kwargs.get('step', kwargs.get('T', 4)),
        img_size=kwargs.get('img_size', 128),
        patch_size=kwargs.get('patch_size', 4),
        in_channels=kwargs.get('in_channels', 2),
        num_classes=kwargs.get('num_classes', 101),
        embed_dim=kwargs.get('embed_dim', 256),
        num_heads=kwargs.get('num_heads', 16),
        mlp_ratio=kwargs.get('mlp_ratio', 4),
        attn_scale=kwargs.get('attn_scale', 0.125),
        attn_drop=kwargs.get('attn_drop', 0.0),
        depths=kwargs.get('depths', 2),
    )
    model.default_cfg = _cfg()
    return model
