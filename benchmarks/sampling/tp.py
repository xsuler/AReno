"""torchrun --standalone --nproc-per-node=2 benchmarks/sampling/tp.py."""
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from areno.engine.parallel.context import TPContext, set_tp_context
from areno.engine.parallel.collectives import all_gather_last_dim
from areno.engine.data import SamplingParams, sampling


def main():
    rank = int(os.environ['LOCAL_RANK']); world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank); dist.init_process_group('nccl')
    set_tp_context(TPContext(rank, world, torch.device('cuda', rank), dist.group.WORLD))
    records = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        torch.manual_seed(634)
        full = torch.randn(8, 131072, dtype=dtype, device='cuda')
        shard = full.chunk(world, dim=-1)[rank].contiguous()
        gathered = all_gather_last_dim(shard)
        torch.testing.assert_close(gathered, full, atol=0, rtol=0)
        params = SamplingParams(temperature=.7, top_k=50, top_p=.9, seed=634)
        tokens = sampling._sample_full_vocab(shard, params, full.shape[1], world, full.device,
                   generator=sampling._make_sample_generator(params, full.device), eos_token_id=None, sample_step=0)
        actual = sampling._policy_token_logprobs(shard, tokens)
        expected = full.double().log_softmax(-1).gather(1, tokens[:, None]).squeeze(-1)
        error = (actual.double() - expected).abs()
        torch.testing.assert_close(actual.double(), expected, atol=3e-6, rtol=1e-6)
        # Correctness above is synchronized. Measure only the collective here.
        for _ in range(10): all_gather_last_dim(shard)
        torch.cuda.synchronize()
        a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(100): all_gather_last_dim(shard)
        b.record(); b.synchronize()
        records.append({'tp':world, 'dtype':str(dtype), 'batch':8, 'vocab':131072,
                        'all_gather_ms':a.elapsed_time(b)/100, 'max_logprob_abs_diff':error.max().item(),
                        'mean_logprob_abs_diff':error.mean().item()})
    if rank == 0:
        print('TP634_RESULT',json.dumps(records),flush=True)
        Path(f'/tmp/qrita634-tp{world}.json').write_text(json.dumps(records, indent=2))
    dist.destroy_process_group()


if __name__ == '__main__': main()
