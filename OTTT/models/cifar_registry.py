"""Shared lazy-construction registry for CIFAR OTTT and BPTT entry points."""
from models import spiking_vgg
from models.spikformer_bn_cifar import online_spikformer_cifar
from models.qkformer_bn_cifar import online_qkformer_cifar
from models.spikformer_diff_cifar import online_spikformer_diff_cifar, online_spikformer_diffv7_full_cifar
from models.qkformer_diff_cifar import online_qkformer_diff_cifar, online_qkformer_diffv7_full_cifar

MODEL_FACTORIES = {
    name: factory for name, factory in vars(spiking_vgg).items()
    if name.startswith('online_') and callable(factory)
}
for alias, factory in (
    ('spikformer', online_spikformer_cifar),
    ('qkformer', online_qkformer_cifar),
    ('spikformer_diff', online_spikformer_diff_cifar),
    ('qkformer_diff', online_qkformer_diff_cifar),
    ('spikformer_diffv7_full', online_spikformer_diffv7_full_cifar),
    ('qkformer_diffv7_full', online_qkformer_diffv7_full_cifar),
):
    MODEL_FACTORIES[alias] = factory
    MODEL_FACTORIES[factory.__name__] = factory
# Explicit version aliases; existing short names remain usable.
MODEL_FACTORIES['spikformer_diffv7'] = online_spikformer_diff_cifar
MODEL_FACTORIES['qkformer_diffv7'] = online_qkformer_diff_cifar
