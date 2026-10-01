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
    ap = argparse.ArgumentParser(description="Sweep decode-only paged attention kernel configs")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--decode_steps", type=int, default=64)
    ap.add_argument("--profile_decode_steps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--decode_block_ns", default="32,64,128")
    ap.add_argument("--decode_num_warps", default="4")
    ap.add_argument("--decode_num_stages", default="2")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    repo = "/home/nano-pointllm"
    py = "/opt/conda/envs/pointllm/bin/python"
    decode_block_ns = [int(x.strip()) for x in args.decode_block_ns.split(",") if x.strip()]
    decode_num_warps = [int(x.strip()) for x in args.decode_num_warps.split(",") if x.strip()]
    decode_num_stages = [int(x.strip()) for x in args.decode_num_stages.split(",") if x.strip()]

    results = []
    with tempfile.TemporaryDirectory(prefix="nanopointllm_attn_sweep_") as td:
        tmpdir = Path(td)
        for block_n in decode_block_ns:
            for num_warps in decode_num_warps:
                for num_stages in decode_num_stages:
                    env = os.environ.copy()
                    env["PYTHONPATH"] = f"/home/PointLLM:{repo}"
                    env["NANOPOINTLLM_ENABLE_CUDA_GRAPH"] = "1"
                    env["NANOPOINTLLM_DECODE_BLOCK_N"] = str(block_n)
                    env["NANOPOINTLLM_DECODE_NUM_WARPS"] = str(num_warps)
                    env["NANOPOINTLLM_DECODE_NUM_STAGES"] = str(num_stages)

                    profile_json = tmpdir / (
                        f"decode_block_n_{block_n}_warps_{num_warps}_stages_{num_stages}_profile.json"
                    )
                    subprocess.run(
                        [
                            py,
                            f"{repo}/scripts/profile_lightweight_decode.py",
                            "--model_path",
                            args.model_path,
                            "--device",
                            args.device,
                            "--batch_size",
                            str(args.batch_size),
                            "--decode_steps",
                            str(args.profile_decode_steps),
                            "--out_json",
                            str(profile_json),
                        ],
                        cwd=repo,
                        env=env,
                        check=True,
                        text=True,
                        capture_output=True,
                    )
                    profile = _read_json(profile_json)

                    bench_json = tmpdir / (
                        f"decode_block_n_{block_n}_warps_{num_warps}_stages_{num_stages}_bench.json"
                    )
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
                    totals = profile.get("totals_ms", {})
                    results.append(
                        {
                            "decode_block_n": block_n,
                            "decode_num_warps": num_warps,
                            "decode_num_stages": num_stages,
                            "replay_tokens_per_sec": paged["replay_tokens_per_sec"],
                            "decode_tokens_per_sec": paged["tokens_per_sec"],
                            "decode_mean_per_step_ms": paged["decode_mean_per_step_ms"],
                            "replay_decode_mean_per_step_ms": paged["replay_decode_mean_per_step_ms"],
                            "attn_ms": round(float(totals.get("attn_ms", 0.0)), 4),
                            "pre_attn_ms": round(float(totals.get("pre_attn_ms", 0.0)), 4),
                            "mlp_ms": round(float(totals.get("mlp_ms", 0.0)), 4),
                            "profile_json": str(profile_json),
                            "bench_json": str(bench_json),
                        }
                    )

    results.sort(key=lambda row: row["replay_tokens_per_sec"], reverse=True)
    payload = {
        "batch_size": args.batch_size,
        "decode_steps": args.decode_steps,
        "profile_decode_steps": args.profile_decode_steps,
        "results": results,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
