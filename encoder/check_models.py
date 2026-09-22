"""CPU checks using real Torch/SpikingJelly; no dataset download or training run."""
import argparse
import ast
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from spikingjelly.clock_driven import functional
from timm.models import create_model, model_entrypoint
from encoder import ENCODINGS, Encoder
from encoder.models import MODEL_NAMES


def check_encodings():
    x = torch.tensor([0., .25, .5, .75, 1.]).view(1, 1, 1, 5)
    ttfs = Encoder(4, 'ttfs')(x).flatten(1)
    expected = torch.tensor([[0., 0., 0., 0., 1.], [0., 0., 0., .5, 0.],
                             [0., 0., 1/3, 0., 0.], [0., .25, 0., 0., 0.]])
    torch.testing.assert_close(ttfs, expected)
    phase = Encoder(10, 'phase')(x)
    torch.testing.assert_close(phase[:2], phase[8:])
    torch.testing.assert_close(phase[:8].sum(0).flatten(), torch.tensor([0., .5, 1., 1.5, 255/128]))
    # Independent scalar reference over every quantized intensity, including 1.
    values = torch.arange(257, dtype=torch.float64).div(256).view(1, 1, 1, -1)
    expected_phase = torch.tensor([[(min(i, 255) >> (7-t%8) & 1) * 2.**(-(t%8))
                                    for i in range(257)] for t in range(17)], dtype=torch.float64)
    torch.testing.assert_close(Encoder(17, 'phase')(values).flatten(1), expected_phase)
    for steps in (1, 4, 8, 11):
        actual = Encoder(steps, 'ttfs')(values).flatten(1)
        expected = torch.zeros_like(actual)
        for t in range(steps):
            for i in range(257):
                if steps-t-1 < i/256*steps <= steps-t:
                    expected[t, i] = 1/(t+1)
        torch.testing.assert_close(actual, expected)
    rate = Encoder(10000, 'rate')(x, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(rate.mean(0), x, rtol=0, atol=.02)
    assert ((rate == 0) | (rate == 1)).all()
    torch.testing.assert_close(rate, Encoder(10000, 'rate')(x, generator=torch.Generator().manual_seed(42)))
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        for kind in ENCODINGS:
            result = Encoder(4, kind)(x.to(dtype))
            expected_dtype = torch.float32 if kind == 'rate' else dtype
            assert result.dtype == expected_dtype and result.is_contiguous() and result.shape == (4, 1, 1, 1, 5)
    # Only TTFS/phase undo normalization; rate must use the supplied tensor.
    for kind in ENCODINGS:
        raw = torch.tensor([-.2, 0., .25, .5, .75, 1., 1.2]).view(1, 1, 1, -1)
        normalized = (raw - .5) / .25
        torch.manual_seed(5)
        actual = Encoder(4, kind, [.5], [.25])(normalized)
        torch.manual_seed(5)
        expected = Encoder(4, kind)(normalized if kind in ('direct', 'rate') else raw.clamp(0, 1))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for bad_step in (0, -1, True, 1.5):
        try:
            Encoder(bad_step)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid step accepted')
    print('PASS encoding references, endpoints, rate statistics, repeatability, dtype and normalization', flush=True)


def check_braincog_rate_parity():
    # BrainCog rate() verbatim expression; compare values, output dtype, RNG
    # consumption and no-grad even when mean/std are passed by the trainer.
    original_default = torch.get_default_dtype()
    try:
        for default_dtype in (torch.float32, torch.float64):
            torch.set_default_dtype(default_dtype)
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                inputs = torch.linspace(-2, 2, 1024, dtype=dtype).reshape(2, 1, 16, 32)
                inputs.requires_grad_()
                for mean, std in ((None, None), ([.4914], [.247])):
                    torch.manual_seed(42)
                    with torch.no_grad():
                        expected = (inputs > torch.rand((8,) + inputs.shape, device=inputs.device)).float()
                    expected_rng = torch.get_rng_state()
                    torch.manual_seed(42)
                    actual = Encoder(8, 'rate', mean, std)(inputs)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    assert actual.dtype == torch.float32 and not actual.requires_grad
                    assert torch.equal(torch.get_rng_state(), expected_rng)
    finally:
        torch.set_default_dtype(original_default)
    print('PASS BrainCog rate exact parity: 4 input dtypes, 2 default dtypes, mean/std, RNG and no-grad', flush=True)


def check_models():
    for name in MODEL_NAMES:
        assert model_entrypoint(name).__module__ == 'encoder.models'
    for classes in (10, 100):
        for name in MODEL_NAMES:
            for kind in ENCODINGS:
                model = create_model(name, backend='torch', encode_type=kind,
                                     num_classes=classes, embed_dims=32, num_heads=4,
                                     mlp_ratios=2, depths=4, T=4)
                signals = []
                stem_inputs = []
                handle = model.encoder.register_forward_hook(lambda m, args, out: signals.append(out.detach()))
                first_conv = next(m for m in model.modules() if isinstance(m, torch.nn.Conv2d))
                pre = first_conv.register_forward_pre_hook(lambda m, args: stem_inputs.append(args[0].detach()))
                x = torch.rand(2, 3, 32, 32)
                output = model(x)
                assert output.shape == (2, classes) and torch.isfinite(output).all()
                assert len(signals) == len(stem_inputs) == 1
                torch.testing.assert_close(stem_inputs[0], signals[0].flatten(0, 1), rtol=0, atol=0)
                handle.remove()
                pre.remove()
                torch.nn.functional.cross_entropy(output, torch.tensor([0, classes-1])).backward()
                assert first_conv.weight.grad is not None and torch.isfinite(first_conv.weight.grad).all()
                assert first_conv.weight.grad.abs().sum() > 0
                assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all()
                for param_name, parameter in model.named_parameters():
                    # Upstream spiking blocks define norm1/norm2 but do not use them.
                    if '.norm1.' in param_name or '.norm2.' in param_name:
                        continue
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), param_name
                model.eval()
                functional.reset_net(model)
                with torch.no_grad():
                    torch.manual_seed(9)
                    first = model(x)
                    functional.reset_net(model)
                    torch.manual_seed(9)
                    second = model(x)
                    torch.testing.assert_close(first, second, rtol=0, atol=0)
                print(f'PASS CIFAR{classes}/{name}/{kind}: encoded stem input, forward/backward, reset', flush=True)
    for name in MODEL_NAMES:
        model = create_model(name, backend='torch', encode_type='phase', num_classes=100).eval()
        with torch.no_grad():
            result = model(torch.rand(1, 3, 32, 32))
        assert result.shape == (1, 100) and torch.isfinite(result).all()
        print(f'PASS default width384/T4/{name}', flush=True)


def check_source_parity(root):
    # Extract reference classes without importing conflicting timm registrations.
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode
    from timm.models.layers import to_2tuple, trunc_normal_, DropPath
    sources = {'qk_gsap': ('gsap/qk.py', 'spiking_transformer'),
               'spiking_gsap': ('gsap/spiking.py', 'vit_snn'),
               'spiking_baseline': ('baseline/spiking.py', 'vit_snn')}
    for name, (path, cls) in sources.items():
        path = root / 'cifar10' / path
        tree = ast.parse(path.read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == 'backend':
                node.value = ast.Constant('torch')
        scope = dict(torch=torch, nn=torch.nn, MultiStepLIFNode=MultiStepLIFNode,
                     to_2tuple=to_2tuple, trunc_normal_=trunc_normal_, DropPath=DropPath)
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), scope)
        kwargs = dict(img_size_h=32, img_size_w=32, patch_size=4, in_channels=3,
                      num_classes=10, embed_dims=32, num_heads=4, mlp_ratios=2,
                      depths=4, T=4, sr_ratios=1)
        reference = scope[cls](**kwargs)
        model = create_model(name, backend='torch', encode_type='direct', **kwargs)
        model.load_state_dict(reference.state_dict(), strict=True)
        x = torch.randn(2, 3, 32, 32)
        expected, actual = reference(x), model(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.square().sum().backward()
        actual.square().sum().backward()
        for (key, a), (key2, b) in zip(reference.named_parameters(), model.named_parameters()):
            assert key == key2
            if a.grad is None:
                assert b.grad is None
            else:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        print(f'PASS original direct output/gradient/checkpoint parity: {name}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    check_encodings()
    check_braincog_rate_parity()
    check_source_parity(args.source_root)
    check_models()
