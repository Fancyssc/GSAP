"""Spikingformer + Diffusion V7; all non-attention code is shared with baseline."""
from functools import partial
from models.spikformer_bn_cifar import vit_snn as Baseline
from models.diffusion_attention import OTTTDiffusionAttentionV7, SpikingDiffusionAttentionV7

__all__ = ['online_spikformer_diff_cifar', 'online_spikformer_diffv7_full_cifar']


class vit_snn(Baseline):
    def __init__(self, img_size_h=32, img_size_w=32,
                 attention_factory=OTTTDiffusionAttentionV7, **kwargs):
        feature_size = (max(img_size_h, img_size_w) + 3) // 4
        super().__init__(attention_factory=partial(
            attention_factory, feature_size=feature_size), **kwargs)


def online_spikformer_diff_cifar(pretrained=False, progress=True,
                                 single_step_neuron=None, **kwargs):
    in_channels = kwargs.pop('in_channels', 3)
    in_channels = kwargs.pop('c_in', in_channels)
    kwargs.setdefault('embed_dims', 384)
    kwargs.setdefault('depths', 4)
    return vit_snn(in_channels=in_channels,
                   single_step_neuron=single_step_neuron, **kwargs)


def online_spikformer_diffv7_full_cifar(**kwargs):
    return online_spikformer_diff_cifar(
        attention_factory=SpikingDiffusionAttentionV7, **kwargs)
