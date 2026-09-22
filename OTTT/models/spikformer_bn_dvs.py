
import torch
import torch.nn as nn
from torch.autograd import Function
from modules.neuron import PPPropLIFNode, Replace, WrapedSNNOp, reset_etrace


__all__ =  [
    "online_spikformer_dvs",
]

'''
    The Conv-LIF structure of the original Spikingformer is adjusted so that
    every conv in the Spikingformer is followed by a LIF, keeping its gradient
    well-defined.
'''


def call_op(op, x, require_wrap):
    if isinstance(op, WrapedSNNOp):
        return op(x, require_wrap=require_wrap)
    return op(x)


def take_spike(x, require_wrap):
    if require_wrap:
        return x[: x.shape[0] // 2]
    return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        single_step_neuron: callable = None,
        **kwargs,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Every conv is immediately followed by a BN, whose mean subtraction
        # cancels the bias, so all convs in this file are built with bias=False.
        self.mlp1_conv = nn.Conv2d(
            in_features, hidden_features, kernel_size=1, stride=1, bias=False
        )
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)
        self.mlp1_lif = single_step_neuron(**kwargs)

        self.mlp2_conv = nn.Conv2d(
            hidden_features, out_features, kernel_size=1, stride=1, bias=False
        )
        self.mlp2_bn = nn.BatchNorm2d(out_features)
        self.mlp2_lif = single_step_neuron(**kwargs)

        if kwargs.get("grad_with_rate", False):
            self.mlp1_conv = WrapedSNNOp(self.mlp1_conv, self.mlp1_lif)
            self.mlp2_conv = WrapedSNNOp(self.mlp2_conv, self.mlp2_lif)

    def forward(self, x, **kwargs):
        require_wrap = kwargs.get("require_wrap", False)

        x = call_op(self.mlp1_conv, x, require_wrap)
        x = self.mlp1_bn(x)
        x = self.mlp1_lif(x, **kwargs)

        x = call_op(self.mlp2_conv, x, require_wrap)
        x = self.mlp2_bn(x)
        x = self.mlp2_lif(x, **kwargs)
        return x


class SpikingSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qk_scale=None,
        single_step_neuron: callable = None,
        **kwargs,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim {dim} must be divisible by num_heads {num_heads}."
            )

        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125 if qk_scale is None else qk_scale

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = single_step_neuron(**kwargs)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = single_step_neuron(**kwargs)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = single_step_neuron(**kwargs)

        self.attn_lif = single_step_neuron(v_threshold=0.5, **kwargs)
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = single_step_neuron(**kwargs)

        if kwargs.get("grad_with_rate", False):
            self.q_conv = WrapedSNNOp(self.q_conv, self.q_lif)
            self.k_conv = WrapedSNNOp(self.k_conv, self.k_lif)
            self.v_conv = WrapedSNNOp(self.v_conv, self.v_lif)
            self.proj_conv = WrapedSNNOp(self.proj_conv, self.proj_lif)
        # attn_lif is driven by the parameter-free matmul, so there is no weight to hook

    def forward(self, x, **kwargs):
        require_wrap = kwargs.get("require_wrap", False)
        _, C, H, W = x.shape

        # The residual stream already carries spikes, so no leading neuron here.
        x = x.flatten(2)
        B, C, N = x.shape

        q_conv_out = call_op(self.q_conv, x, require_wrap)
        q_conv_out = self.q_bn(q_conv_out)
        q_conv_out = self.q_lif(q_conv_out, **kwargs)
        q = q_conv_out.transpose(-1, -2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        k_conv_out = call_op(self.k_conv, x, require_wrap)
        k_conv_out = self.k_bn(k_conv_out)
        k_conv_out = self.k_lif(k_conv_out, **kwargs)
        k = k_conv_out.transpose(-1, -2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        v_conv_out = call_op(self.v_conv, x, require_wrap)
        v_conv_out = self.v_bn(v_conv_out)
        v_conv_out = self.v_lif(v_conv_out, **kwargs)
        v = v_conv_out.transpose(-1, -2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        x = (attn @ v) * self.scale

        x = x.transpose(2, 3).reshape(B, C, N)
        x = self.attn_lif(take_spike(x, require_wrap), **kwargs)
        x = call_op(self.proj_conv, x, require_wrap)
        x = self.proj_bn(x).reshape(-1, C, H, W)
        x = self.proj_lif(x, **kwargs)
        return x


class SpikingTransformer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qk_scale=None,
        single_step_neuron: callable = None,
        **kwargs,
    ):
        super().__init__()
        self.attn = SpikingSelfAttention(
            dim,
            num_heads=num_heads,
            qk_scale=qk_scale,
            single_step_neuron=single_step_neuron,
            **kwargs,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            single_step_neuron=single_step_neuron,
            **kwargs,
        )

    def forward(self, x, **kwargs):
        x = x + self.attn(x, **kwargs)
        x = x + self.mlp(x, **kwargs)
        return x


class SpikingTokenizer(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embed_dims=384,
        single_step_neuron: callable = None,
        **kwargs,
    ):
        super().__init__()
        self.block0_conv = nn.Conv2d(
            in_channels,
            embed_dims // 8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.block0_bn = nn.BatchNorm2d(embed_dims // 8)
        self.block0_lif = single_step_neuron(**kwargs)

        self.block1_conv = nn.Conv2d(
            embed_dims // 8,
            embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.block1_bn = nn.BatchNorm2d(embed_dims // 4)
        self.block1_mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1_lif = single_step_neuron(**kwargs)

        self.block2_conv = nn.Conv2d(
            embed_dims // 4,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.block2_bn = nn.BatchNorm2d(embed_dims // 2)
        self.block2_mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block2_lif = single_step_neuron(**kwargs)

        self.block3_conv = nn.Conv2d(
            embed_dims // 2,
            embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.block3_bn = nn.BatchNorm2d(embed_dims)
        self.block3_mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block3_lif = single_step_neuron(**kwargs)

        self.block4_conv = nn.Conv2d(
            embed_dims,
            embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.block4_bn = nn.BatchNorm2d(embed_dims)
        self.block4_mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block4_lif = single_step_neuron(**kwargs)

        if kwargs.get("grad_with_rate", False):
            if isinstance(self.block1_lif, PPPropLIFNode):
                for index in range(1, 5):
                    conv = getattr(self, f"block{index}_conv")
                    lif = getattr(self, f"block{index}_lif")
                    post = nn.Sequential(
                        getattr(self, f"block{index}_bn"),
                        getattr(self, f"block{index}_mp"),
                    )
                    setattr(self, f"block{index}_conv",
                            WrapedSNNOp(conv, lif, post_op=post))
            else:
                for index in range(1, 5):
                    conv = getattr(self, f"block{index}_conv")
                    setattr(self, f"block{index}_conv", WrapedSNNOp(conv))
        if isinstance(self.block0_lif, PPPropLIFNode):
            # block0_conv consumes the raw image, not spikes, so there is no rate to
            # reroute (OTTT leaves it unwrapped), but it still drives block0_lif, so
            # under pp-prop it is still wrapped to obtain eps_f
            self.block0_conv = WrapedSNNOp(self.block0_conv, self.block0_lif, wrap=False)

    def forward(self, x, **kwargs):
        require_wrap = kwargs.get("require_wrap", False)
        if x.ndim != 4:
            raise ValueError(
                "SpikingTokenizer expects [B, C, H, W], "
                f"but received shape {tuple(x.shape)}."
            )

        # block0_conv consumes the raw image, which is constant across time steps, so
        # there is no pre-synaptic trace to reroute (wrap=False); under pp-prop it is
        # still wrapped, because it still drives block0_lif and still needs eps_f.
        x = call_op(self.block0_conv, x, require_wrap)
        x = self.block0_bn(x)
        x = self.block0_lif(x, **kwargs)

        x = call_op(self.block1_conv, x, require_wrap)
        if not getattr(self.block1_conv, "includes_post_op", False):
            x = self.block1_bn(x)
            x = self.block1_mp(x)
        x = self.block1_lif(x, **kwargs)

        x = call_op(self.block2_conv, x, require_wrap)
        if not getattr(self.block2_conv, "includes_post_op", False):
            x = self.block2_bn(x)
            x = self.block2_mp(x)
        x = self.block2_lif(x, **kwargs)

        x = call_op(self.block3_conv, x, require_wrap)
        if not getattr(self.block3_conv, "includes_post_op", False):
            x = self.block3_bn(x)
            x = self.block3_mp(x)
        x = self.block3_lif(x, **kwargs)

        x = call_op(self.block4_conv, x, require_wrap)
        if not getattr(self.block4_conv, "includes_post_op", False):
            x = self.block4_bn(x)
            x = self.block4_mp(x)
        x = self.block4_lif(x, **kwargs)

        return x # + x_feat


class vit_snn(nn.Module):
    def __init__(
        self,
        in_channels=2,
        num_classes=10,
        embed_dims=256,
        num_heads=8,
        mlp_ratios=4.0,
        qk_scale=None,
        depths=4,
        single_step_neuron: callable = None,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.single_step_neuron = single_step_neuron
        self.grad_with_rate = kwargs.get("grad_with_rate", False)

        self.patch_embed = SpikingTokenizer(
            in_channels=in_channels,
            embed_dims=embed_dims,
            single_step_neuron=single_step_neuron,
            **kwargs,
        )
        self.block = nn.ModuleList(
            [
                SpikingTransformer(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qk_scale=qk_scale,
                    single_step_neuron=single_step_neuron,
                    **kwargs,
                )
                for _ in range(depths)
            ]
        )

        self.head = (
            nn.Linear(embed_dims, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.apply(self._init_weights)
        # Wrapped last, so _init_weights still sees a bare nn.Linear. The head
        # now follows a neuron, so it has a trace and gets the same OTTT
        # treatment as the classifier in the VGG and QKFormer models.
        if self.grad_with_rate and num_classes > 0:
            self.head = WrapedSNNOp(self.head)  # readout: drives no neuron

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward_features(self, x, **kwargs):
        x = self.patch_embed(x, **kwargs)
        for block in self.block:
            x = block(x, **kwargs)
        return x.flatten(2).mean(2)

    def forward(self, x, **kwargs):
        if x.ndim != 4:
            raise ValueError(
                "vit_snn expects one time step in [B, C, H, W] layout, "
                f"but received shape {tuple(x.shape)}."
            )

        if kwargs.get("init", False):  # weights run before the neurons, so clear eps_f at the start of the sequence here
            reset_etrace(self)
        require_wrap = self.grad_with_rate and self.training
        kwargs = dict(kwargs)
        kwargs["require_wrap"] = require_wrap
        if require_wrap:
            kwargs["output_type"] = "spike_rate"

        x = self.forward_features(x, **kwargs)
        return call_op(self.head, x, require_wrap)

    def get_spike(self):
        spikes = []
        if self.single_step_neuron is None:
            return spikes
        for module in self.modules():
            if isinstance(module, self.single_step_neuron) and hasattr(
                module, "spike"
            ):
                spike = module.spike.cpu()
                spikes.append(spike.reshape(spike.shape[0], -1))
        return spikes


def _spikingformer(embed_dims, depths, single_step_neuron: callable = None, **kwargs):
    """Shared body of the size-specific factories, mirroring ``_qkformer``."""

    num_classes = kwargs.pop("num_classes", 10)
    in_channels = kwargs.pop("in_channels", 3)
    in_channels = kwargs.pop("c_in", in_channels)
    return vit_snn(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dims=embed_dims,
        depths=depths,
        single_step_neuron=single_step_neuron,
        **kwargs,
    )


def online_spikformer_dvs(
    pretrained=False, progress=True, single_step_neuron: callable = None, **kwargs
):
    return _spikingformer(
        embed_dims=256, depths=2, single_step_neuron=single_step_neuron, **kwargs
    )


def online_spikingformer(
    pretrained=False, progress=True, single_step_neuron: callable = None, **kwargs
):
    """Default size, kept so existing callers keep working."""

    return _spikingformer(
        embed_dims=256, depths=2, single_step_neuron=single_step_neuron, **kwargs
    )
