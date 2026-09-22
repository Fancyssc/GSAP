from typing import Callable, overload
import torch
import torch.nn as nn
from . import surrogate
from .neuron_spikingjelly import IFNode, LIFNode

class OnlineIFNode(IFNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = None,
            surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = True,
            track_rate: bool = True, neuron_dropout: float = 0.0, **kwargs):

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.track_rate = track_rate
        self.dropout = neuron_dropout
        if self.track_rate:
            self.register_memory('rate_tracking', None)
        if self.dropout > 0.0:
            self.register_memory('mask', None)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v.detach() + x

    # should be initialized at the first time step
    def forward_init(self, x: torch.Tensor):
        self.v = torch.zeros_like(x)
        self.rate_tracking = None
        if self.dropout > 0.0 and self.training:
            self.mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            self.mask = self.mask.requires_grad_(False) / (1 - self.dropout)

    def forward(self, x: torch.Tensor, **kwargs):
        init = kwargs.get('init', False)
        save_spike = kwargs.get('save_spike', False)
        output_type = kwargs.get('output_type', 'spike')
        if init:
            self.forward_init(x)

        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        if self.dropout > 0.0 and self.training:
            spike = self.mask.expand_as(spike) * spike

        if save_spike:
            self.spike = spike

        if self.track_rate:
            with torch.no_grad():
                if self.rate_tracking == None:
                    self.rate_tracking = spike.clone().detach()
                else:
                    self.rate_tracking = self.rate_tracking + spike.clone().detach()

        if output_type == 'spike_rate':
            assert self.track_rate == True
            return torch.cat((spike, self.rate_tracking), dim=0)
        else:
            return spike


class OnlineLIFNode(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
            v_reset: float = None, surrogate_function: Callable = surrogate.Sigmoid(),
            detach_reset: bool = True, track_rate: bool = True, neuron_dropout: float = 0.0, **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.track_rate = track_rate
        self.dropout = neuron_dropout
        if self.track_rate:
            self.register_memory('rate_tracking', None)
        if self.dropout > 0.0:
            self.register_memory('mask', None)

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            self.v = self.v.detach() * (1 - 1. / self.tau) + x
        else:
            self.v = self.v.detach() * (1 - 1. / self.tau) + self.v_reset / self.tau + x

    # should be initialized at the first time step
    def forward_init(self, x: torch.Tensor):
        self.v = torch.zeros_like(x)
        self.rate_tracking = None
        if self.dropout > 0.0 and self.training:
            self.mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            self.mask = self.mask.requires_grad_(False) / (1 - self.dropout)

    def forward(self, x: torch.Tensor, **kwargs):
        init = kwargs.get('init', False)
        save_spike = kwargs.get('save_spike', False)
        output_type = kwargs.get('output_type', 'spike')
        if init:
            self.forward_init(x)

        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        if self.dropout > 0.0 and self.training:
            spike = self.mask.expand_as(spike) * spike

        if save_spike:
            self.spike = spike

        if self.track_rate:
            with torch.no_grad():
                if self.rate_tracking == None:
                    self.rate_tracking = spike.clone().detach()
                else:
                    self.rate_tracking = self.rate_tracking * (1 - 1. / self.tau) + spike.clone().detach()


        # this branch returns cat([spike, rate])
        if output_type == 'spike_rate':
            assert self.track_rate == True
            return torch.cat((spike, self.rate_tracking), dim=0)
        else:
            return spike


class BPTTLIFNode(LIFNode):
    """Plain single-step LIF neuron, as in SpikingJelly, for BPTT training.

    The membrane dynamics (``neuronal_charge`` / ``neuronal_fire`` /
    ``neuronal_reset``) are inherited unchanged from :class:`LIFNode`, so ``v``
    keeps its graph across time steps and one backward at the end of the T
    steps back-propagates through time.

    Unlike :class:`OnlineLIFNode` there is no rate tracking: the eligibility
    trace only exists to build OTTT's instantaneous gradient, while BPTT gets
    the temporal credit assignment from the graph itself.

    The only additions over the SpikingJelly neuron are the ``kwargs``-based
    call signature the models in this repo use (``init`` resets ``v`` at the
    first time step, ``save_spike`` keeps the spikes for ``get_spike()``) and
    the optional per-neuron dropout.
    """

    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
            v_reset: float = None, surrogate_function: Callable = surrogate.Sigmoid(),
            detach_reset: bool = True, neuron_dropout: float = 0.0, **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.dropout = neuron_dropout
        if self.dropout > 0.0:
            self.register_memory('mask', None)

    # should be initialized at the first time step
    def forward_init(self, x: torch.Tensor):
        self.v = torch.zeros_like(x)
        if self.dropout > 0.0 and self.training:
            self.mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            self.mask = self.mask.requires_grad_(False) / (1 - self.dropout)

    def forward(self, x: torch.Tensor, **kwargs):
        init = kwargs.get('init', False)
        save_spike = kwargs.get('save_spike', False)
        assert kwargs.get('output_type', 'spike') == 'spike', \
            'BPTTLIFNode does not track rates; build the model with grad_with_rate=False'
        if init:
            self.forward_init(x)

        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        if self.dropout > 0.0 and self.training:
            spike = self.mask.expand_as(spike) * spike

        if save_spike:
            self.spike = spike

        return spike


class ScaleGrad(torch.autograd.Function):
    """Identity in the forward pass; backward multiplies the gradient element-wise by ``scale`` (which must already be detached)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor):
        ctx.save_for_backward(scale)
        return x

    @staticmethod
    def backward(ctx, grad):
        scale, = ctx.saved_tensors
        return grad * scale, None


class PPPropLIFNode(OnlineLIFNode):

    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
            v_reset: float = None, surrogate_function: Callable = surrogate.Sigmoid(),
            detach_reset: bool = False, track_rate: bool = True, neuron_dropout: float = 0.0,
            etrace_decay: float = None, **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function,
                         detach_reset, track_rate, neuron_dropout, **kwargs)
        self.etrace_decay = (1. - 1. / tau) if etrace_decay is None else etrace_decay
        assert 0. <= self.etrace_decay < 1., \
            f'etrace_decay should be in [0, 1). While we got {self.etrace_decay}.'
        # post-synaptic trace, and the reset Jacobian carried over from the previous step
        self.register_memory('etrace_f', None)
        self.register_memory('etrace_reset_jac', None)
        # whether eps_f is applied by this neuron itself. Set to False when the weight
        # driving it is attached (it is then applied by the weight instead; see
        # WrapedSNNOp.pprop_op for the rationale).
        self.etrace_at_neuron = True
        # set on the second pass: the weight output and this neuron's state are not the
        # same set of neurons; see update_etrace_f
        self.etrace_incompatible = False

    def etrace_reset(self):
        """Drop the trace so the next step starts a new sequence.

        Called by :func:`reset_etrace` from the model's top-level forward;
        see the notes there.
        """
        self.etrace_f = None

    def reset_jacobian(self) -> torch.Tensor:
        """du^{t+1}/du^t at the current membrane potential, excluding the leak.

        Must be called after charging and before reset, when ``v`` is still the
        pre-reset u^t. The surrogate gradient is derived from the surrogate
        function on the fly rather than hard-coded, so any function in
        :mod:`modules.surrogate` can be used directly.
        """
        if self.detach_reset:          # reset carries no gradient, D^t reduces to the leak
            return torch.ones_like(self.v)
        with torch.enable_grad():
            u = (self.v - self.v_threshold).detach().requires_grad_(True)
            spike = self.surrogate_function(u)
            sg = torch.autograd.grad(spike.sum(), u)[0]
        if self.v_reset is None:       # soft reset, u <- u - V_th * s
            return 1. - self.v_threshold * sg
        else:                          # hard reset, u <- (1 - s) * u + v_reset * s
            return (1. - spike.detach()) - sg * (self.v - self.v_reset)

    def update_etrace_f(self, ref: torch.Tensor) -> torch.Tensor:
        """Advance eps_f by one step and return the scale to apply to the gradient of
        ``ref``; returns None when not applicable.

        ``ref`` only provides shape/dtype/device. Charging has already multiplied
        the gradient by the constant ``D_f``, so the return value divides it back
        out, leaving ``eps_f``.
        """
        if self.etrace_incompatible:
            return None
        df = 1. / self.tau if self.decay_input else 1.
        if self.etrace_f is None:
            self.etrace_f = torch.zeros_like(ref)
            self.etrace_reset_jac = torch.zeros_like(ref)
        # D^t: hidden-to-hidden Jacobian from the previous step, leak times the reset term
        d_hid = (1. - 1. / self.tau) * self.etrace_reset_jac
        if d_hid.shape != self.etrace_f.shape:
            if d_hid.numel() == self.etrace_f.numel():
                d_hid = d_hid.reshape(self.etrace_f.shape)   # only a view in between
            else:
                # There is a shape-changing path between the weight and the neuron that
                # is not yet covered by post_op. Attaching it to the neuron would
                # penalize all upstream weights, so keep this safe fallback and revert
                # to the instantaneous gradient.
                print(f'PPPropLIFNode: eps_f disabled, the weight outputs '
                      f'{tuple(self.etrace_f.shape)} but the neuron holds '
                      f'{tuple(d_hid.shape)}')
                self.etrace_incompatible = True
                self.etrace_f = None
                return None
        self.etrace_f = self.etrace_decay * d_hid * self.etrace_f + (1. - self.etrace_decay) * df
        return self.etrace_f / df

    def neuronal_charge(self, x: torch.Tensor):
        # the trace only serves the backward pass; skipped entirely under no_grad
        # (eval, firing-rate statistics), same cost as OnlineLIFNode
        if not torch.is_grad_enabled():
            return super().neuronal_charge(x)
        if self.etrace_at_neuron:
            with torch.no_grad():
                scale = self.update_etrace_f(x)
            if scale is not None:
                x = ScaleGrad.apply(x, scale)
        super().neuronal_charge(x)
        with torch.no_grad():
            # while v is still the pre-reset potential, save the reset Jacobian for
            # the next step's D^{t+1}
            self.etrace_reset_jac = self.reset_jacobian()


class Replace(torch.autograd.Function):
    """Forward takes the value of ``z1_r``; backward routes the gradient to both inputs (OTTT)."""

    @staticmethod
    def forward(ctx, z1, z1_r):
        return z1_r

    @staticmethod
    def backward(ctx, grad):
        return grad, grad


class WrapedSNNOp(nn.Module):
    """Operator wrapper that runs the forward on spikes and reroutes the weight
    gradient onto the rate.

    ``neuron`` is the neuron driven by this weight and selects the branch:

    * ``None`` — OTTT. Also used for readouts that drive no neuron.
    * :class:`PPPropLIFNode` — pp-prop. On top of OTTT, multiplies this neuron's
      post-synaptic trace eps_f onto **this weight's** gradient.

    ``wrap=False`` is for the first conv, which consumes the raw image rather
    than spikes: it has no rate to reroute to, but still drives a neuron and
    still needs eps_f.

    ``post_op`` is the stateless/instantaneous path between the weight and the
    neuron (e.g. BN + Pool). pp-prop must apply eps_f after ``post_op`` and
    before the neuron, then map back to this weight through ``post_op``'s
    Jacobian; otherwise the shapes differ across pooling and MaxPool's argmax
    routing would be lost. OTTT does not use this parameter and keeps the
    original call graph unchanged.
    """

    def __init__(self, op: nn.Module, neuron: nn.Module = None, wrap: bool = True,
                 post_op: nn.Module = None):
        super(WrapedSNNOp, self).__init__()
        self.op = op
        self.wrap = wrap
        # post_op stays registered under the model's original bn/pool attributes to
        # avoid changing state_dict keys; keep only an unregistered reference here,
        # as for neuron. The model forward skips the outer duplicate call when
        # includes_post_op=True.
        self._post_op = [post_op]

        if isinstance(neuron, PPPropLIFNode):
            neuron.etrace_at_neuron = False   # eps_f is now applied by this weight
            self._neuron = [neuron]
            self.op_for_grad = self.pprop_op
        else:
            self._neuron = [None]
            self.op_for_grad = self.ottt_op

    @property
    def neuron(self):
        return self._neuron[0]

    @property
    def post_op(self):
        return self._post_op[0]

    @property
    def includes_post_op(self):
        return self.post_op is not None

    @staticmethod
    def _functional_call(module: nn.Module, x: torch.Tensor, detach_params: bool):
        """Run the gradient branch without letting BN-like modules update running buffers twice."""
        params = {
            name: (parameter.detach() if detach_params else parameter)
            for name, parameter in module.named_parameters()
        }
        buffers = {
            name: buffer.detach().clone()
            for name, buffer in module.named_buffers()
        }
        return torch.func.functional_call(module, (params, buffers), (x,))

    def ottt_op(self, x: torch.Tensor, require_wrap: bool) -> torch.Tensor:
        """OTTT: forward on spikes; on the gradient path the input is replaced by the rate, so the weight gradient is the outer product with eps_x."""
        if not require_wrap:
            return self.op(x)
        assert x.shape[0] % 2 == 0, \
            f'expected cat([spike, rate], dim=0), but got batch dimension {x.shape[0]}'
        B = x.shape[0] // 2
        spike, rate = x[:B], x[B:]
        with torch.no_grad():
            out = self.op(spike).detach()            # forward truth uses the spike
        in_for_grad = Replace.apply(spike, rate)     # gradient path swaps in the rate, i.e. eps_x
        return Replace.apply(self.op(in_for_grad), out)

    def pprop_op(self, x: torch.Tensor, require_wrap: bool) -> torch.Tensor:
        """pp-prop: the same rate rerouting as OTTT, plus applying eps_f only to
        this weight's gradient.

        Attached to the neuron, eps_f would sit on the shared backward path and
        every upstream weight would take a share: layer l would end up scaled by
        prod_{k>=l} eps_f^k. Autograd cannot split one tensor's gradient into
        two destinations, so the operator forks it into two branches: ``y_x``
        detaches its parameters, its cotangent reaches only the input,
        unscaled; ``y_w`` detaches its input, scales its cotangent by eps_f,
        and contributes only the weight gradient. The cost is one extra forward
        through the operator.
        """
        # keep the original implementation when there is no composite
        # post-processing, to avoid changing the existing conv/linear PPProp numerics.
        if self.post_op is None:
            return self._plain_pprop_op(x, require_wrap)

        if not torch.is_grad_enabled():
            return self.post_op(self.op(x))

        if require_wrap:
            assert x.shape[0] % 2 == 0, \
                f'expected cat([spike, rate], dim=0), but got batch dimension {x.shape[0]}'
            B = x.shape[0] // 2
            spike, rate = x[:B], x[B:]
            actual_input = spike
            x_for_grad = Replace.apply(spike, rate)
        else:
            actual_input = x
            x_for_grad = x

        # the real forward runs only once, so BN running stats update only once.
        # The two functional branches below use cloned buffers, build only local
        # VJPs, and leave module state untouched.
        with torch.no_grad():
            actual_core = self.op(actual_input).detach()
            out = self.post_op(actual_core).detach()

        core_x = self._functional_call(self.op, x_for_grad, detach_params=True)
        core_x = Replace.apply(core_x, actual_core)
        y_x = self._functional_call(self.post_op, core_x, detach_params=True)

        with torch.no_grad():
            scale = self.neuron.update_etrace_f(y_x)

        core_w = self._functional_call(self.op, x_for_grad.detach(), detach_params=False)
        core_w = Replace.apply(core_w, actual_core)
        # post_op's parameters (mainly BN gamma/beta) are not synaptic weights
        # scaled by eps_f in this repo's original implementation; detach them so
        # the Jacobian only routes the scaled cotangent back to op.
        y_w = self._functional_call(self.post_op, core_w, detach_params=True)
        if scale is not None:
            y_w = ScaleGrad.apply(y_w, scale)

        # BN-like post_op parameters keep their original plain gradient; their
        # input is cut so this third branch does not redundantly affect op/x.
        y_post = self._functional_call(
            self.post_op, actual_core.detach(), detach_params=False)

        # y_x: unscaled upstream input gradient; y_w: op parameter gradient scaled
        # by eps_f; y_post: unscaled BN parameter gradient. All three branches keep
        # their forward value pinned to the real spike path.
        y = (y_x
             + y_w - y_w.detach()
             + y_post - y_post.detach())
        return Replace.apply(y, out)

    def _plain_pprop_op(self, x: torch.Tensor, require_wrap: bool) -> torch.Tensor:
        """Original PPProp path when there is no BN/Pool post-processing."""
        if require_wrap:
            assert x.shape[0] % 2 == 0, \
                f'expected cat([spike, rate], dim=0), but got batch dimension {x.shape[0]}'
            B = x.shape[0] // 2
            spike, rate = x[:B], x[B:]
            with torch.no_grad():
                out = self.op(spike).detach()
            x = Replace.apply(spike, rate)

        if torch.is_grad_enabled():
            params = {name: p.detach() for name, p in self.op.named_parameters()}
            y_x = torch.func.functional_call(self.op, params, (x,))
            with torch.no_grad():
                scale = self.neuron.update_etrace_f(y_x)
        else:
            scale = None

        if scale is None:
            y = self.op(x)
        else:
            y_w = ScaleGrad.apply(self.op(x.detach()), scale)
            y = y_x + y_w - y_w.detach()

        return Replace.apply(y, out) if require_wrap else y

    def forward(self, x: torch.Tensor, **kwargs):
        return self.op_for_grad(x, self.wrap and kwargs.get('require_wrap', True))


def reset_etrace(net: nn.Module):
    """Clear all post-synaptic traces in ``net`` so the next step starts a new
    sequence.

    Must run before the first weight of the step, which is why the model's
    top-level ``forward`` calls it on ``init=True`` rather than the neurons'
    own ``forward_init`` — at t=0 the weights run first and would already have
    read the previous sequence's leftover traces.
    """
    for m in net.modules():
        if isinstance(m, PPPropLIFNode):
            m.etrace_reset()
