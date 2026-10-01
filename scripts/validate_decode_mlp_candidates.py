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
    ap = argparse.ArgumentParser(description="Validate top decode-MLP sweep candidates on longer replay")
    ap.add_argument("--candidates_json", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--decode_steps", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    repo = "/home/nano-pointllm"
    py = "/opt/conda/envs/pointllm/bin/python"
    candidates = _read_json(Path(args.candidates_json)).get("results", [])[: args.top_k]

    results = []
    with tempfile.TemporaryDirectory(prefix="nanopointllm_candidate_validate_") as td:
        tmpdir = Path(td)
        for idx, row in enumerate(candidates):
            env = os.environ.copy()
            env["PYTHONPATH"] = f"/home/PointLLM:{repo}"
            env["NANOPOINTLLM_ENABLE_CUDA_GRAPH"] = "1"
            env["NANOPOINTLLM_GATEUP_INPUT_LAYOUT"] = row.get("gateup_input_layout", "auto")
            env["NANOPOINTLLM_DOWNPROJ_INPUT_LAYOUT"] = row.get("downproj_input_layout", "auto")
            env["NANOPOINTLLM_GATEUP_SPLIT_IMPL"] = row.get("gateup_split_impl", "split")
            env["NANOPOINTLLM_GATEUP_WEIGHT_LAYOUT"] = row.get("gateup_weight_layout", "split")
            env["NANOPOINTLLM_GATEUP_GEMM_IMPL"] = row.get("gateup_gemm_impl", "linear")
            env["NANOPOINTLLM_DOWNPROJ_GEMM_IMPL"] = row.get("downproj_gemm_impl", "linear")

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
                    "source_tag": row.get("tag", f"candidate_{idx}"),
                    "gateup_input_layout": env["NANOPOINTLLM_GATEUP_INPUT_LAYOUT"],
                    "downproj_input_layout": env["NANOPOINTLLM_DOWNPROJ_INPUT_LAYOUT"],
                    "gateup_split_impl": env["NANOPOINTLLM_GATEUP_SPLIT_IMPL"],
                    "gateup_weight_layout": env["NANOPOINTLLM_GATEUP_WEIGHT_LAYOUT"],
                    "gateup_gemm_impl": env["NANOPOINTLLM_GATEUP_GEMM_IMPL"],
                    "downproj_gemm_impl": env["NANOPOINTLLM_DOWNPROJ_GEMM_IMPL"],
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
