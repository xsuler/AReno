"""Run from the repo: python benchmarks/sampling/pivot.py --output results.json."""

import argparse
import json
import platform
import importlib.util
import subprocess
import time
from pathlib import Path

import torch
from areno.accel.kernels.sampling import truncate_probs
from areno.engine.data import SamplingParams, sampling


def legacy(p, k, top_p):
    v, ids = p.sort(descending=True)
    if k > 0:
        v[:, min(k, p.shape[1]) :] = 0
    if top_p < 1:
        v[(v.cumsum(-1) - v) > top_p] = 0
    v = sampling._sanitize_sorted_probs(v)
    return ids.gather(-1, torch.multinomial(v, 1)).squeeze(-1)


def measure(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    torch.cuda.reset_peak_memory_stats()
    initial = torch.cuda.memory_allocated()
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 20)
    return {
        "ms": samples,
        "median_ms": sorted(samples)[2],
        "peak_temporary_bytes": torch.cuda.max_memory_allocated() - initial,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("sampling_baseline", Path(__file__).with_name("baseline.py"))
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    results = {
        "commit": Path("SOURCE_COMMIT").read_text().strip()
        if Path("SOURCE_COMMIT").exists()
        else subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "rows": [],
    }
    for batch in (1, 8, 32):
        for vocab in (32768, 131072, 151936):
            p = torch.randn(batch, vocab, device="cuda").softmax(-1)
            for k, top_p in ((50, 0.9), (0, 0.9), (50, 1.0), (0, 1.0)):
                filtered = k > 0 or top_p < 1
                functions = {
                    "baseline": lambda: legacy(p, k, top_p) if filtered else torch.multinomial(p, 1),
                    "pivot": lambda: torch.multinomial(truncate_probs(p, k, top_p), 1)
                    if filtered
                    else torch.multinomial(p, 1),
                }
                record = {"batch": batch, "vocab": vocab, "k": k, "p": top_p}
                for name, fn in functions.items():
                    record[name] = measure(fn)
                logits = p.log()
                params = SamplingParams(temperature=0.8, top_k=k, top_p=top_p)
                record["full_baseline"] = measure(lambda: baseline._sample(logits, params, logits.device))
                record["full_pivot"] = measure(lambda: sampling._sample(logits, params, logits.device))
                if filtered:
                    record["filter_only"] = measure(lambda: truncate_probs(p, k, top_p))
                results["rows"].append(record)
                Path(args.output).write_text(json.dumps(results, indent=2))
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
