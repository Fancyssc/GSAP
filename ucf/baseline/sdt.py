import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg


class SPS(nn.Module):
    def __init__(self, step=4, img_h=128, img_w=128, patch_size=4, in_channels=2, embed_dims=384, backend='cupy', **kwargs):
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
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

    def forward(self, x):
        T, B, C, H, W = x.shape

        # print(x.shape)

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H , W).contiguous()
        x = self.proj_lif(x).flatten(0,1).contiguous()
        x = self.maxpool(x)
        # print(x.shape)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()
        x = self.maxpool1(x)
        # print(x.shape)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)
        # print(x.shape)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        x = self.maxpool3(x).reshape(T, B, -1, H // 16, W // 16).contiguous()
        # print(x.shape)
        x_feat = x
        x = self.proj_lif3(x)

        x = self.rpe_conv(x.flatten(0, 1))
        x = self.rpe_bn(x).reshape(T, B, -1, H // 16, W // 16).contiguous()
        x = x + x_feat

        return x


class SDSA(nn.Module):
    def __init__(self, embed_dim, step=4, num_heads=12, backend='cupy'):
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

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=backend)

        self.talking_heads = nn.Conv1d(num_heads, num_heads, kernel_size=1, stride=1, bias=False)
        self.talking_heads_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.proj_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(embed_dim)

        self.shortcut_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        identity = x
        x = self.shortcut_lif(x)

        x_for_qkv = x.flatten(0, 1)
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
            .contiguous()
        )
        k = (
            k_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        v = (
            v_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        kv = k.mul(v)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
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
    def __init__(self, in_features, step=4, mlp_ratio=4.0, out_features=None, mlp_drop=0., backend='cupy'):
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
        T, B, C, H, W = x.shape
        identity = x

        x = self.fc_lif1(x)
        x = self.fc_conv1(x.flatten(0, 1))
        x = self.fc_bn1(x).reshape(T, B, -1, H, W)

        x = self.fc_lif2(x)
        x = self.fc_conv2(x.flatten(0, 1))
        x = self.fc_bn2(x).reshape(T, B, -1, H, W)

        return x + identity


class SDT_Block_s(nn.Module):
    def __init__(self, embed_dim=384, num_heads=12, step=4, mlp_ratio=4., mlp_drop=0., backend='cupy'):
        super().__init__()

        self.attn = SDSA(embed_dim, step=step, num_heads=num_heads, backend=backend)
        self.mlp = MLP(step=step, in_features=embed_dim, mlp_ratio=mlp_ratio, out_features=embed_dim, mlp_drop=mlp_drop, backend=backend)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x


class SDTV1(nn.Module):
    def __init__(
        self,
        step=4,
        img_size=128,
        patch_size=4,
        in_channels=2,
        num_classes=101,
        embed_dim=384,
        num_heads=12,
        mlp_ratio=4,
        mlp_drop=0.,
        depths=4,
        backend='cupy',
    ):
        super().__init__()
        self.step = step
        self.num_classes = num_classes
        self.depths = depths
        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        patch_embed = SPS(
            step=step,
            img_h=img_size,
            img_w=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dim,
            backend=backend,
        )

        block = nn.ModuleList([
            SDT_Block_s(
                step=step,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                mlp_drop=mlp_drop,
                backend=backend,
            )
            for _ in range(depths)
        ])

        self.patch_embed = patch_embed
        self.block = block
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
        x = self.patch_embed(x)
        for blk in self.block:
            x = blk(x)
        T, B, C, H, W = x.shape
        x = x.flatten(0, 1).flatten(-2, -1)

        return x.mean(2).reshape(T, B, C).contiguous()

    def forward(self, x):
        x = x.transpose(0, 1).contiguous()
        x = self.forward_features(x)
        x = self.head_lif(x)
        x = self.head(x).mean(0)
        return x


@register_model
def sdt_baseline(pretrained=False, **kwargs):
    if pretrained:
        raise ValueError('No pretrained weights are provided for sdt_baseline')
    model = SDTV1(
        step=kwargs.get('step', kwargs.get('T', 4)),
        img_size=kwargs.get('img_size', 128),
        patch_size=kwargs.get('patch_size', 4),
        in_channels=kwargs.get('in_channels', 2),
        num_classes=kwargs.get('num_classes', 101),
        embed_dim=kwargs.get('embed_dim', 256),
        num_heads=kwargs.get('num_heads', 16),
        mlp_ratio=kwargs.get('mlp_ratio', 4),
        mlp_drop=kwargs.get('mlp_drop', 0.0),
        depths=kwargs.get('depths', 2),
        backend=kwargs.get('backend', 'cupy'),
    )
    model.default_cfg = _cfg(num_classes=model.num_classes,
                             input_size=(kwargs.get('in_channels', 2),
                                         kwargs.get('img_size', 128),
                                         kwargs.get('img_size', 128)))
    return model


@register_model
def SDT(pretrained=False, **kwargs):
    """Compatibility alias for the original model name."""
    return sdt_baseline(pretrained=pretrained, **kwargs)


def _set_lif_backend(model, backend):
    for module in model.modules():
        if isinstance(module, MultiStepLIFNode):
            module.backend = backend


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run SDT with a fake UCF101-DVS input.')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--img-size', type=int, default=128)
    parser.add_argument('--in-channels', type=int, default=2)
    parser.add_argument('--num-classes', type=int, default=101)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=16)
    parser.add_argument('--depths', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = SDT(
        step=args.step,
        img_size=args.img_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depths=args.depths,
    ).to(device)

    if device.type == 'cpu':
        _set_lif_backend(model, 'torch')

    x = torch.rand(
        args.batch_size,
        args.step,
        args.in_channels,
        args.img_size,
        args.img_size,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        y = model(x)

    print(f'input shape:  {tuple(x.shape)}')
    print(f'output shape: {tuple(y.shape)}')
