"""QKFormer + Diffusion V7 at all three stages, sharing the baseline body."""
from models.qkformer_bn_cifar import spiking_transformer as Baseline
from models.diffusion_attention import OTTTDiffusionAttentionV7, SpikingDiffusionAttentionV7

__all__ = ['online_qkformer_diff_cifar', 'online_qkformer_diffv7_full_cifar']


class spiking_transformer(Baseline):
    def __init__(self, attention_factory=OTTTDiffusionAttentionV7, **kwargs):
        super().__init__(attention_factory=attention_factory, **kwargs)


def online_qkformer_diff_cifar(pretrained=False, progress=True,
                              single_step_neuron=None, **kwargs):
    in_channels = kwargs.pop('in_channels', 3)
    in_channels = kwargs.pop('c_in', in_channels)
    kwargs.setdefault('embed_dims', 384)
    kwargs.setdefault('depths', 4)
    return spiking_transformer(in_channels=in_channels,
                              single_step_neuron=single_step_neuron, **kwargs)


def online_qkformer_diffv7_full_cifar(**kwargs):
    return online_qkformer_diff_cifar(
        attention_factory=SpikingDiffusionAttentionV7, **kwargs)
