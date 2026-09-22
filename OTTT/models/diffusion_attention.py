"""OTTT-native diffusion and the full V7 reference implementation."""
import torch.nn as nn
from modules.neuron import OnlineLIFNode, PPPropLIFNode, WrapedSNNOp
from models.spikformer_bn_cifar import call_op, take_spike


class OTTTDiffusionAttentionV7(nn.Module):
    """Parallel local/axial mixing of the incoming spike/eligibility stream.

    Each spatial convolution consumes the original presynaptic trace. Currents
    are summed before BN/LIF; that LIF builds the trace for the output projection.
    Axial branches cover a cross, not full 2-D propagation within one block.
    """
    def __init__(self, dim, num_heads=None, feature_size=8,
                 single_step_neuron=None, input_is_spike=True, **kwargs):
        super().__init__()
        single_step_neuron = single_step_neuron or OnlineLIFNode
        if issubclass(single_step_neuron, PPPropLIFNode):
            raise ValueError('Diffusion V7 supports OTTT and BPTT; PPProp is not implemented')
        if not input_is_spike:
            raise ValueError('OTTT V7 requires spike/trace input; use *_diffv7_full for analog input')
        if not isinstance(feature_size, int) or feature_size < 1:
            raise ValueError('feature_size must be a positive integer')
        self.feature_size = feature_size
        kernel = 2 * feature_size - 1
        self.local_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.row_conv = nn.Conv2d(dim, dim, (1, kernel),
                                  padding=(0, feature_size - 1), groups=dim, bias=False)
        self.column_conv = nn.Conv2d(dim, dim, (kernel, 1),
                                     padding=(feature_size - 1, 0), groups=dim, bias=False)
        self.attn_bn = nn.BatchNorm2d(dim)
        self.attn_lif = single_step_neuron(**kwargs)
        self.proj_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(dim)
        self.output_lif = single_step_neuron(**kwargs)
        if kwargs.get('grad_with_rate', False):
            for name in ('local_conv', 'row_conv', 'column_conv', 'proj_conv'):
                setattr(self, name, WrapedSNNOp(getattr(self, name)))

    def forward(self, x, **kwargs):
        require_wrap = kwargs.get('require_wrap', False)
        if x.ndim != 4 or max(x.shape[-2:]) > self.feature_size:
            raise ValueError(f'V7 expects [B,C,H,W] with H,W <= {self.feature_size}')
        local = call_op(self.local_conv, x, require_wrap)
        row = call_op(self.row_conv, x, require_wrap)
        column = call_op(self.column_conv, x, require_wrap)
        current = take_spike(x, require_wrap) + local + row + column
        spike = self.attn_lif(self.attn_bn(current), **kwargs)
        current = self.proj_bn(call_op(self.proj_conv, spike, require_wrap))
        return self.output_lif(current, **kwargs)


class SpikingDiffusionAttentionV7(nn.Module):
    """Local + gated row/column diffusion, with a spiking output adapter.

    The shared OTTT backbone supplies sums of spikes and their eligibility
    traces. Consume them directly, like baseline attention: another input LIF
    would clip overlapping residual spikes and replace the incoming trace.
    Set input_is_spike=False for analog inputs / the previous port's input LIF.

    All internal V7 branches are retained. output_lif converts the analog
    projection to this backbone's spike/rate residual interface. BN and the
    instantaneous product see only the physical batch, never traces.
    """
    def __init__(self, dim, num_heads=None, feature_size=8,
                 single_step_neuron=None, input_is_spike=True, **kwargs):
        super().__init__()
        single_step_neuron = single_step_neuron or OnlineLIFNode
        if issubclass(single_step_neuron, PPPropLIFNode):
            raise ValueError('Diffusion V7 supports OTTT and BPTT; PPProp is not implemented')
        if not isinstance(feature_size, int) or feature_size < 1:
            raise ValueError('feature_size must be a positive integer')
        self.feature_size = feature_size
        kernel = 2 * feature_size - 1

        def lif(threshold=None):
            options = dict(kwargs)
            if threshold is not None:
                options['v_threshold'] = threshold
            return single_step_neuron(**options)

        self.proj_lif = None if input_is_spike else lif()
        self.feature_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.feature_bn = nn.BatchNorm2d(dim)
        self.feature_lif = lif()
        self.local_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.local_bn = nn.BatchNorm2d(dim)
        self.row_conv = nn.Conv2d(dim, dim, (1, kernel), padding=(0, feature_size - 1), groups=dim, bias=False)
        self.row_bn = nn.BatchNorm2d(dim)
        self.row_lif = lif()
        self.column_conv = nn.Conv2d(dim, dim, (kernel, 1), padding=(feature_size - 1, 0), groups=dim, bias=False)
        self.column_bn = nn.BatchNorm2d(dim)
        self.gate_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.gate_bn = nn.BatchNorm2d(dim)
        self.gate_lif = lif(0.5)
        self.attn_bn = nn.BatchNorm2d(dim)
        self.attn_lif = lif(0.5)
        self.proj_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(dim)
        self.output_lif = lif()
        if kwargs.get('grad_with_rate', False):
            for name in ('feature_conv', 'local_conv', 'row_conv',
                         'column_conv', 'gate_conv', 'proj_conv'):
                setattr(self, name, WrapedSNNOp(getattr(self, name)))

    def forward(self, x, **kwargs):
        require_wrap = kwargs.get('require_wrap', False)
        if x.ndim != 4 or max(x.shape[-2:]) > self.feature_size:
            raise ValueError(f'V7 expects [B,C,H,W] with H,W <= {self.feature_size}')
        # Preserve both halves of the residual stream for the three synaptic
        # branches. Only the optional analog-input adapter builds a new trace.
        spike = x if self.proj_lif is None else self.proj_lif(
            take_spike(x, require_wrap), **kwargs)
        local = self.local_bn(call_op(self.local_conv, spike, require_wrap))
        gate = self.gate_bn(call_op(self.gate_conv, spike, require_wrap))
        gate = self.gate_lif(gate, **dict(kwargs, output_type='spike'))
        feature = self.feature_bn(call_op(self.feature_conv, spike, require_wrap))
        feature = self.feature_lif(feature, **kwargs)
        message = self.row_bn(call_op(self.row_conv, feature, require_wrap))
        message = self.row_lif(message + take_spike(feature, require_wrap), **kwargs)
        message = self.column_bn(call_op(self.column_conv, message, require_wrap))
        x = self.attn_bn(take_spike(spike, require_wrap) + local + message * gate)
        x = self.attn_lif(x, **kwargs)
        x = self.proj_bn(call_op(self.proj_conv, x, require_wrap))
        return self.output_lif(x, **kwargs)
