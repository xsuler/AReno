"""Audit actual rollout logits; intentionally synchronizes, never a timing run.

python benchmarks/sampling/audit.py --model LOCAL_PATH --tp 2 --output DIR
"""

import argparse
import json
import os
from pathlib import Path

import torch

from areno import Trainer
from areno.api import CudaConfig, SamplingParams
from areno.engine import inference
from areno.engine.parallel.collectives import all_gather_last_dim
from areno.engine.parallel.context import get_tp_context

_original_logprobs = inference._policy_token_logprobs


def checked_logprobs(logits_shard, tokens):
    actual = _original_logprobs(logits_shard, tokens)
    full = all_gather_last_dim(logits_shard)
    expected = full.double().log_softmax(-1).gather(-1, tokens[..., None]).squeeze(-1)
    torch.testing.assert_close(actual.double(), expected, atol=3e-6, rtol=1e-6)
    error = (actual.double() - expected).abs()
    record = {
        "rank": get_tp_context().rank,
        "count": error.numel(),
        "max_abs_diff": error.max().item(),
        "mean_abs_diff": error.mean().item(),
    }
    with (Path(os.environ["ARENO_AUDIT_DIR"]) / f"rank{record['rank']}.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return actual


# Spawned workers import this script too, installing the hook before inference.
inference._policy_token_logprobs = checked_logprobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["ARENO_AUDIT_DIR"] = str(Path(args.output).resolve())
    Path(args.output).mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        args.tp,
        args.model,
        custom_config=CudaConfig(
            tp_size=args.tp, devices=list(range(args.tp)), runtime={"compile_model": False, "eager_decode": True}
        ),
    )
    trainer.init()
    try:
        params = SamplingParams(temperature=0.7, top_k=50, top_p=0.9, max_new_tokens=4, ignore_eos=True, seed=634)
        out = trainer.rollout_batch(["Explain why the sky is blue."] * 2, 1, params)
        assert all(len(r.sequences[0].resp_tokens) == 4 for r in out)
        print("ROLLOUT_LOGPROB_AUDIT_OK", flush=True)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
