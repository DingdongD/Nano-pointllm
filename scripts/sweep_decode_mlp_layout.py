#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep decode-only MLP layout variants")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--decode_steps", type=int, default=64)
    ap.add_argument("--profile_decode_steps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--gateup_layouts", default="auto,contiguous,clone")
    ap.add_argument("--downproj_layouts", default="auto,contiguous,clone")
    ap.add_argument("--split_impls", default="split,slice")
    ap.add_argument("--gateup_weight_layouts", default="split")
    ap.add_argument("--gateup_gemm_impls", default="linear")
    ap.add_argument("--downproj_gemm_impls", default="linear")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    repo = "/home/nano-pointllm"
    py = "/opt/conda/envs/pointllm/bin/python"
    gateup_layouts = tuple(x.strip() for x in args.gateup_layouts.split(",") if x.strip())
    downproj_layouts = tuple(x.strip() for x in args.downproj_layouts.split(",") if x.strip())
    split_impls = tuple(x.strip() for x in args.split_impls.split(",") if x.strip())
    gateup_weight_layouts = tuple(x.strip() for x in args.gateup_weight_layouts.split(",") if x.strip())
    gateup_gemm_impls = tuple(x.strip() for x in args.gateup_gemm_impls.split(",") if x.strip())
    downproj_gemm_impls = tuple(x.strip() for x in args.downproj_gemm_impls.split(",") if x.strip())

    results = []
    with tempfile.TemporaryDirectory(prefix="nanopointllm_layout_sweep_") as td:
        tmpdir = Path(td)
        for gateup_layout, downproj_layout, split_impl, gateup_weight_layout, gateup_gemm_impl, downproj_gemm_impl in itertools.product(
            gateup_layouts,
            downproj_layouts,
            split_impls,
            gateup_weight_layouts,
            gateup_gemm_impls,
            downproj_gemm_impls,
        ):
            tag = (
                f"g-{gateup_layout}_d-{downproj_layout}_s-{split_impl}"
                f"_gw-{gateup_weight_layout}"
                f"_gg-{gateup_gemm_impl}_dg-{downproj_gemm_impl}"
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = f"/home/PointLLM:{repo}"
            env["NANOPOINTLLM_ENABLE_CUDA_GRAPH"] = "1"
            env["NANOPOINTLLM_GATEUP_INPUT_LAYOUT"] = gateup_layout
            env["NANOPOINTLLM_DOWNPROJ_INPUT_LAYOUT"] = downproj_layout
            env["NANOPOINTLLM_GATEUP_SPLIT_IMPL"] = split_impl
            env["NANOPOINTLLM_GATEUP_WEIGHT_LAYOUT"] = gateup_weight_layout
            env["NANOPOINTLLM_GATEUP_GEMM_IMPL"] = gateup_gemm_impl
            env["NANOPOINTLLM_DOWNPROJ_GEMM_IMPL"] = downproj_gemm_impl

            profile_json = tmpdir / f"{tag}_profile.json"
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
                env=env,
                cwd=repo,
                check=True,
                text=True,
                capture_output=True,
            )
            profile = _read_json(profile_json)

            bench_json = tmpdir / f"{tag}_bench.json"
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
                    "tag": tag,
                    "gateup_input_layout": gateup_layout,
                    "downproj_input_layout": downproj_layout,
                    "gateup_split_impl": split_impl,
                    "gateup_weight_layout": gateup_weight_layout,
                    "gateup_gemm_impl": gateup_gemm_impl,
                    "downproj_gemm_impl": downproj_gemm_impl,
                    "replay_tokens_per_sec": paged["replay_tokens_per_sec"],
                    "decode_tokens_per_sec": paged["tokens_per_sec"],
                    "mlp_gateup_ms": round(float(totals.get("mlp_gateup_ms", 0.0)), 4),
                    "mlp_act_mul_ms": round(float(totals.get("mlp_act_mul_ms", 0.0)), 4),
                    "mlp_downproj_staging_ms": round(float(totals.get("mlp_downproj_staging_ms", 0.0)), 4),
                    "mlp_downproj_ms": round(float(totals.get("mlp_downproj_ms", 0.0)), 4),
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
    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
