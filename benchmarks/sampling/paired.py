"""Same-engine sampler crossover; benchmark-only hooks run in spawned workers.

Seed 635 selects legacy, but the generator is seeded with 634 in both modes.
No production configuration or source code is changed by this script.
"""

import argparse
import importlib.util
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

from areno import Trainer
from areno.api import CudaConfig, SamplingParams
from areno.engine import inference
from areno.engine.data import sampling

_spec = importlib.util.spec_from_file_location("sampling_legacy", Path(__file__).with_name("baseline.py"))
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
_original_sample = sampling._sample
_original_generator = sampling._make_sample_generator


def sample(logits, params, device, **kwargs):
    fn = _legacy._sample if params.seed == 635 else _original_sample
    return fn(logits, params, device, **kwargs)


def generator(params, device):
    return _original_generator(replace(params, seed=634) if params.seed == 635 else params, device)


sampling._sample = sample
sampling._make_sample_generator = generator
inference._make_sample_generator = generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    trainer = Trainer(
        args.tp,
        args.model,
        custom_config=CudaConfig(
            tp_size=args.tp,
            devices=list(range(args.tp)),
            runtime={"compile_model": False, "decode_graph_buckets": [1, 8]},
        ),
    )
    trainer.init()
    trainer.begin_rollout_session()
    records = []
    try:
        for k, p in ((50, 0.9), (0, 0.9), (0, 1.0)):
            record = {"k": k, "p": p, "tp": args.tp, "batch": 8, "times": {}}
            last = {}
            for length in (16, 64):
                times = {"baseline": [], "pivot": []}

                def run(mode):
                    params = SamplingParams(
                        temperature=0.8,
                        top_k=k,
                        top_p=p,
                        seed=635 if mode == "baseline" else 634,
                        max_new_tokens=length,
                        ignore_eos=True,
                    )
                    start = time.perf_counter()
                    out = trainer.rollout_batch(["Explain why the sky is blue in simple words."] * 8, 1, params)
                    elapsed = time.perf_counter() - start
                    tokens = [r.sequences[0].resp_tokens for r in out]
                    assert all(len(row) == length for row in tokens)
                    last[mode] = tokens
                    return elapsed

                for _ in range(3):
                    run("baseline")
                    run("pivot")
                for iteration in range(5):
                    order = ("baseline", "pivot") if iteration % 2 == 0 else ("pivot", "baseline")
                    for mode in order:
                        times[mode].append(run(mode))
                if k == 0 and p == 1.0:
                    assert last["baseline"] == last["pivot"], "unfiltered control token streams differ"
                record["times"][str(length)] = times
            for mode in ("baseline", "pivot"):
                tpot = (
                    statistics.median(record["times"]["64"][mode]) - statistics.median(record["times"]["16"][mode])
                ) / 48
                record[mode] = {"incremental_tpot_ms": 1000 * tpot, "incremental_tokens_per_second": 8 / tpot}
            records.append(record)
            Path(args.output).write_text(json.dumps(records, indent=2))
            print("PAIRED634", json.dumps(record), flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        trainer.end_rollout_session()
        trainer.close()


if __name__ == "__main__":
    main()
