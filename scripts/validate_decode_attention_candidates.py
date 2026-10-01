#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate top decode-attention sweep candidates on longer replay")
    ap.add_argument("--candidates_json", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--decode_steps", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    repo = "/home/nano-pointllm"
    py = "/opt/conda/envs/pointllm/bin/python"
    candidates = _read_json(Path(args.candidates_json)).get("results", [])[: args.top_k]

    results = []
    with tempfile.TemporaryDirectory(prefix="nanopointllm_attn_validate_") as td:
        tmpdir = Path(td)
        for idx, row in enumerate(candidates):
            env = os.environ.copy()
            env["PYTHONPATH"] = f"/home/PointLLM:{repo}"
            env["NANOPOINTLLM_ENABLE_CUDA_GRAPH"] = "1"
            env["NANOPOINTLLM_DECODE_BLOCK_N"] = str(row.get("decode_block_n", 64))
            env["NANOPOINTLLM_DECODE_NUM_WARPS"] = str(row.get("decode_num_warps", 4))
            env["NANOPOINTLLM_DECODE_NUM_STAGES"] = str(row.get("decode_num_stages", 2))

            bench_json = tmpdir / f"candidate_{idx}.json"
            subprocess.run(
                [
                    py,
                    f"{repo}/scripts/bench_paged_pointllm.py",
                    "--model_path",
                    args.model_path,
                    "--device",
                    args.device,
                    "--batch_sizes",
                    str(args.batch_size),
                    "--decode_steps",
                    str(args.decode_steps),
                    "--warmup",
                    str(args.warmup),
                    "--runs",
                    str(args.runs),
                    "--out_json",
                    str(bench_json),
                ],
                cwd=repo,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            bench = _read_json(bench_json)
            paged = next(item for item in bench["paged_results"] if item["mode"] == "paged")
            results.append(
                {
                    "rank_from_source": idx + 1,
                    "decode_block_n": int(env["NANOPOINTLLM_DECODE_BLOCK_N"]),
                    "decode_num_warps": int(env["NANOPOINTLLM_DECODE_NUM_WARPS"]),
                    "decode_num_stages": int(env["NANOPOINTLLM_DECODE_NUM_STAGES"]),
                    "decode_tokens_per_sec": paged["tokens_per_sec"],
                    "replay_tokens_per_sec": paged["replay_tokens_per_sec"],
                    "decode_mean_per_step_ms": paged["decode_mean_per_step_ms"],
                    "replay_decode_mean_per_step_ms": paged["replay_decode_mean_per_step_ms"],
                }
            )

    results.sort(key=lambda row: row["replay_tokens_per_sec"], reverse=True)
    payload = {
        "source_json": args.candidates_json,
        "batch_size": args.batch_size,
        "decode_steps": args.decode_steps,
        "results": results,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
