"""Differential support/probability tests; no cross-algorithm RNG equality."""
import pytest
import torch

from areno.engine.data import SamplingParams, sampling

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(p, k, top_p):
    values, ids = p.sort(descending=True, stable=True)
    if k > 0:
        values[:, k:] = 0
    if top_p < 1:
        # FP64 exclusive sums avoid cancellation in cumsum(values)-values.
        prefix = values.double().cumsum(-1) - values.double()
        values[prefix > top_p] = 0
    values /= values.sum(-1, keepdim=True)
    return torch.zeros_like(p).scatter(1, ids, values)


@pytest.mark.parametrize('vocab', [32, 32768, 131072, 151936])
@pytest.mark.parametrize('k,p', [(1, 1.), (50, .9), (0, .9), (0, 0.), (999999, 1.), (0, 1e-6)])
def test_probabilities(vocab, k, p):
    from areno.accel.kernels.sampling import truncate_probs
    torch.manual_seed(634)
    probs = torch.randn(4, vocab, device='cuda').softmax(-1)
    probs[1].fill_(1 / vocab)
    probs[2].zero_(); probs[2, 0] = 1
    expected = reference(probs, k, p)
    actual = truncate_probs(probs, k, p)
    assert torch.equal(actual > 0, expected > 0)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_seed_graph_suppression_and_invalid(dtype):
    logits = torch.randn(8, 32768, device='cuda', dtype=dtype)
    logits[0].fill_(float('nan')); logits[1].fill_(-float('inf'))
    logits[2, 2] = float('inf')
    params = SamplingParams(temperature=.7, top_k=50, top_p=.9,
                            suppress_token_ids=(2, 3), min_new_tokens=2, seed=634)
    def run(generator):
        return sampling._sample(logits, params, logits.device, generator=generator,
                                eos_token_id=4, sample_step=0)
    a = run(sampling._make_sample_generator(params, logits.device))
    b = run(sampling._make_sample_generator(params, logits.device))
    assert torch.equal(a, b)
    assert not torch.isin(a, torch.tensor([2, 3, 4], device='cuda')).any()
    for _ in range(3):
        run(None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run(None)
    graph.replay(); torch.cuda.synchronize()
    assert (out >= 0).all() and (out < logits.shape[-1]).all()


def test_distribution():
    from areno.accel.kernels.sampling import truncate_probs
    probs = torch.tensor([[.4, .3, .2, .1]], device='cuda')
    filtered = truncate_probs(probs, 3, .5)
    draws = torch.multinomial(filtered, 100000, replacement=True)
    freq = torch.bincount(draws[0], minlength=4) / 100000
    torch.testing.assert_close(freq, reference(probs, 3, .5)[0], atol=.008, rtol=0)
