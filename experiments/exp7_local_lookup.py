#!/usr/bin/env python3
"""Isolate local lookup from legacy full-context lookup; no speedup claims for an LM."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time

import _bootstrap  # noqa: F401
import torch

from tinyengram import Engram, EngramConfig


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    module = Engram(EngramConfig(
        vocab_size=64, hidden_size=32, hc_mult=1, layer_ids=(0,),
        n_head_per_ngram=2, n_embed_per_ngram=8, slots_per_head=67,
    ), 0).eval()
    rows = []
    for length in [8, 32, 128, 512]:
        ids = torch.randint(0, 64, (1, length))
        for chunk in [1, 4]:
            def legacy():
                # Mirror the old context suffix check as well as its full gather.
                new_ids = module.compress(ids[:, -chunk:])
                context = module.compress(ids)
                if not torch.equal(context[:, -chunk:], new_ids):
                    raise ValueError("input must be the context suffix")
                return module.memory.retrieve(context)[:, -chunk:]
            def local():
                return module.retrieve(ids[:, -chunk:], ids)
            torch.testing.assert_close(legacy(), local())
            for _ in range(5):
                legacy()
                local()
            samples = {"legacy": [], "local": []}
            for iteration in range(args.repeats):
                methods = [("legacy", legacy), ("local", local)]
                if iteration % 2:
                    methods.reverse()
                for name, fn in methods:
                    start = time.perf_counter()
                    fn()
                    samples[name].append((time.perf_counter() - start) * 1e6)
            before, after = (statistics.median(samples[n]) for n in ["legacy", "local"])
            rows.append({"context_length": length, "new_tokens": chunk,
                         "legacy_gather_positions": length, "local_gather_positions": chunk,
                         "local_hash_positions": min(length, chunk+2), "exact_match": True,
                         "legacy_median_us": before, "local_median_us": after,
                         "speedup": before / after, "samples_us": samples})
            print(f"context={length:3} new={chunk} gather={length}->{chunk} "
                  f"lookup={before:.1f}->{after:.1f}us {before / after:.3f}x")
    output = _bootstrap.RUNS / "local-lookup.json"
    output.write_text(json.dumps({"torch": torch.__version__, "device": "cpu", "threads": 1,
        "seed": 42, "repeats": args.repeats, "scope": "Engram retrieve only, not end-to-end",
        "source_sha256": {p: hashlib.sha256((_bootstrap.ROOT / p).read_bytes()).hexdigest()
                          for p in ["solutions/tinyengram/engram.py",
                                    "experiments/exp7_local_lookup.py"]},
        "rows": rows}, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
