#!/usr/bin/env python3
"""Reproducible TinyLM speculative experiment; see docs/experiment-speculative.md."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import _bootstrap  # noqa: F401
import torch

from speculative import greedy_speculative_generate
from recall import recall_batch
from tinyengram import EngramConfig, TinyLM, TinyLMConfig, generate, naive_generate


def build(hidden, layers, max_seq, engram=False, conv_kernel=1):
    ecfg = EngramConfig(
        vocab_size=32, hidden_size=hidden, hc_mult=1, layer_ids=(0,),
        n_head_per_ngram=2, n_embed_per_ngram=8, slots_per_head=67,
        conv_kernel_size=conv_kernel, conv_zero_init=False,
    ) if engram else None
    return TinyLM(TinyLMConfig(
        vocab_size=32, hidden_size=hidden, n_layers=layers, n_heads=4,
        ffn_mult=2, hc_mult=1, max_seq=max_seq, tie_weights=False,
        engram=ecfg, engram_layers=(0,) if engram else (),
    ))


def cyclic_ids(batch, length, generator, device):
    starts = torch.randint(0, 31, (batch, 1), generator=generator)
    return ((starts + torch.arange(length)[None]) % 31 + 1).to(device)


def train(model, steps, seed, task="cycle", train_length=32):
    """Toy next-token cycle, NOT language modeling or long-context recall."""
    rng = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    model.train()
    for _ in range(steps):
        device = next(model.parameters()).device
        if task == "recall":
            length = int(torch.randint(7, train_length + 1, (), generator=rng))
            prompt, response = recall_batch(16, length, rng, device)
            ids = torch.cat([prompt, response], 1)
            labels = ids[:, 1:].clone()
            labels[:, :length-1] = -100  # supervise only the four copied values + EOS
        else:
            ids = cyclic_ids(16, train_length + 1, rng, device)
            labels = ids[:, 1:]
        _, loss, _ = model(ids[:, :-1], targets=labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return {"steps": steps, "first_loss": losses[0], "last_loss": losses[-1],
            "task": task, "train_length": train_length}


def timed(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    value = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, (time.perf_counter() - start) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=["trained", "random", "identical"], default="trained")
    parser.add_argument("--draft-engram", action="store_true")
    parser.add_argument("--conv-kernel", type=int, default=1,
                        help="Engram convolution kernel; 4 exercises temporal state")
    parser.add_argument("--task", choices=["cycle", "recall"], default="cycle")
    parser.add_argument("--train-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--draft-lengths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[4, 32, 96])
    parser.add_argument("--new-tokens", type=int,
                        help="default: cycle=32, recall=5 (four copied values + EOS)")
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=_bootstrap.RUNS / "speculative.json")
    args = parser.parse_args()
    if args.new_tokens is None:
        args.new_tokens = 5 if args.task == "recall" else 32
    if min(args.draft_lengths + args.prompt_lengths + [args.new_tokens, args.prompts,
           args.repeats, args.threads, args.steps, args.conv_kernel, args.train_length]) < 1 \
            or args.warmup < 0:
        parser.error("lengths/counts/steps must be positive; warmup must be nonnegative")
    if args.pair == "identical" and args.draft_engram:
        parser.error("identical is a plain-model control; omit --draft-engram")
    if args.task == "recall" and (min(args.prompt_lengths) < 7 or args.train_length < 7):
        parser.error("recall prompt/train lengths must be >= 7")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    max_seq = max(args.train_length + 5, max(args.prompt_lengths) + args.new_tokens)
    target = build(64, 2, max_seq).to(device)
    draft = build(32, 1, max_seq, args.draft_engram, args.conv_kernel).to(device)
    training = {}
    if args.pair == "trained":
        training["target"] = train(target, args.steps, args.seed + 1,
                                    args.task, args.train_length)
        training["draft"] = train(draft, args.steps, args.seed + 1,
                                   args.task, args.train_length)
    elif args.pair == "identical":
        draft = deepcopy(target)  # explicit maximum-acceptance correctness control
    target.eval()
    draft.eval()
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_bootstrap.ROOT, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    report = {
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform(), "device": str(device),
                        "threads": args.threads, "base_revision": revision},
        "arguments": {**vars(args), "output": str(args.output)},
        "models": {"target_parameters": target.num_parameters,
                   "draft_parameters": draft.num_parameters,
                   "target_config": asdict(target.cfg), "draft_config": asdict(draft.cfg)},
        "training": training,
        "task": ("synthetic cycle 1..31; no long-distance recall required" if args.task == "cycle"
                 else "single-binding four-value copy; random payload, noise and key; "
                      "training prompt lengths 7..train_length; evaluation uses independent RNG"),
        "timing": "milliseconds; prefill + decode + both model/cache setup; excludes training, "
                  "model construction, reference checks and audit hooks; CUDA synchronized",
        "rows": [],
    }
    rng = torch.Generator().manual_seed(args.seed + 2)
    for length in args.prompt_lengths:
        expected = None
        if args.task == "recall":
            prompts, expected = recall_batch(args.prompts, length, rng, device)
        else:
            prompts = cyclic_ids(args.prompts, length, rng, device)
        cases = []
        for prompt in prompts.split(1):
            reference = generate(target, prompt, args.new_tokens)
            assert torch.equal(reference, naive_generate(target, prompt, args.new_tokens))
            cases.append((prompt, reference))
        for k in args.draft_lengths:
            baseline_ms, speculative_ms, summaries, outputs = [], [], [], []
            for case_index, (prompt, reference) in enumerate(cases):
                def baseline():
                    return generate(target, prompt, args.new_tokens)

                def speculative():
                    return greedy_speculative_generate(
                        target, draft, prompt, args.new_tokens, draft_length=k,
                    )
                for _ in range(args.warmup):
                    baseline()
                    speculative()
                for repeat in range(args.repeats):
                    # Alternate execution order to reduce systematic timing bias.
                    if repeat % 2:
                        result, sms = timed(speculative, device)
                        normal, bms = timed(baseline, device)
                    else:
                        normal, bms = timed(baseline, device)
                        result, sms = timed(speculative, device)
                    assert torch.equal(normal, reference)
                    assert torch.equal(result.ids, reference), (length, k)
                    baseline_ms.append(bms)
                    speculative_ms.append(sms)
                summaries.append(result.summary())
                case = {"prompt": prompt[0].tolist(), "output": result.ids[0].tolist(),
                        "rounds": [asdict(r) for r in result.rounds]}
                if expected is not None:
                    count = min(args.new_tokens, 5)
                    with torch.no_grad():
                        draft_only = generate(draft, prompt, count)[:, length:]
                    case.update({
                        "expected_response": expected[case_index].tolist(),
                        "target_response_token_accuracy": float((
                            reference[:, length:length+count] == expected[case_index, :count]
                        ).float().mean()),
                        "draft_response_token_accuracy": float((
                            draft_only == expected[case_index, :count]
                        ).float().mean()),
                    })
                outputs.append(case)
            rounds = sum(s["rounds"] for s in summaries)
            accepted = sum(s["accepted_draft_tokens"] for s in summaries)
            proposed = sum(s["proposed_tokens"] for s in summaries)
            bmed, smed = statistics.median(baseline_ms), statistics.median(speculative_ms)
            row = {
                "prompt_length": length, "draft_length": k, "exact_match": True,
                "mean_accepted_draft_tokens": accepted / rounds,
                "mean_emitted_tokens": args.prompts * args.new_tokens / rounds,
                "acceptance_rate": accepted / proposed,
                "rejection_rounds": sum(s["rejection_rounds"] for s in summaries),
                "baseline_median_ms": bmed, "speculative_median_ms": smed,
                "speedup": bmed / smed,
                "baseline_ms_samples": baseline_ms, "speculative_ms_samples": speculative_ms,
                "per_prompt_summary": summaries, "cases": outputs,
            }
            report["rows"].append(row)
            print(f"P={length:3} K={k} accepted={accepted / rounds:.2f} "
                  f"emitted={row['mean_emitted_tokens']:.2f} "
                  f"AR={row['acceptance_rate']:.1%} "
                  f"baseline={bmed:.2f}ms spec={smed:.2f}ms speedup={bmed / smed:.3f}x")
    # A source fingerprint identifies local changes even before a commit exists.
    report["source_sha256"] = {
        name: hashlib.sha256((_bootstrap.ROOT / name).read_bytes()).hexdigest()
        for name in ["experiments/speculative.py", "experiments/exp6_speculative.py",
                     "experiments/recall.py", "solutions/tinyengram/cache.py",
                     "solutions/tinyengram/engram.py", "solutions/tinyengram/transformer.py"]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
