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
    ap = argparse.ArgumentParser(description="Cleaner decode attention profile summary")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--decode_steps", type=int, default=4)
    ap.add_argument("--exclude_first_n_layers", type=int, default=1)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    repo = "/home/nano-pointllm"
    py = "/opt/conda/envs/pointllm/bin/python"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"/home/PointLLM:{repo}"

    with tempfile.TemporaryDirectory(prefix="nanopointllm_attn_clean_") as td:
        raw_json = Path(td) / "raw_profile.json"
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
                str(args.decode_steps),
                "--out_json",
                str(raw_json),
            ],
            cwd=repo,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        raw = _read_json(raw_json)

    layers = raw.get("layers", [])
    trimmed = layers[args.exclude_first_n_layers :]
    count = max(len(trimmed), 1)
    totals = {}
    means = {}
    for row in trimmed:
        for key, value in row.items():
            if key == "layer":
                continue
            totals[key] = totals.get(key, 0.0) + float(value)
    for key, value in totals.items():
        means[key] = value / count

    payload = {
        "batch_size": args.batch_size,
        "decode_steps": args.decode_steps,
        "exclude_first_n_layers": args.exclude_first_n_layers,
        "num_layers_total": len(layers),
        "num_layers_trimmed": len(trimmed),
        "totals_ms_trimmed": totals,
        "means_ms_trimmed": means,
        "layers_trimmed": trimmed,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
