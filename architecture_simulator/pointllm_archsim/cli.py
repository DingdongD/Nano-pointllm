"""Command-line interface for presets, simulation, evidence, and DSE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dse import SweepSpec, latency_resource_pareto, run_sweep
from .evidence import extract_ncu_evidence
from .report import plot_simulation, plot_sweep, write_rows_csv
from .schema import HardwareConfig, Workload
from .simulator import ArchitectureSimulator, summary_row
from .workload import build_pointllm_7b_workload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preset = subparsers.add_parser("preset", help="write a PointLLM-7B workload trace")
    _workload_arguments(preset)
    preset.add_argument("--output", type=Path, required=True)

    simulate = subparsers.add_parser("simulate", help="run one architecture configuration")
    simulate.add_argument("--config", type=Path, required=True)
    simulate.add_argument("--workload", type=Path)
    _workload_arguments(simulate)
    simulate.add_argument("--output_dir", type=Path, required=True)
    simulate.add_argument("--plot", action="store_true")

    sweep = subparsers.add_parser("sweep", help="run a Cartesian design-space sweep")
    sweep.add_argument("--config", type=Path, required=True)
    sweep.add_argument("--sweep", type=Path, required=True)
    sweep.add_argument("--workload", type=Path)
    _workload_arguments(sweep)
    sweep.add_argument("--output_dir", type=Path, required=True)
    sweep.add_argument("--plot", action="store_true")

    evidence = subparsers.add_parser("extract-evidence", help="normalize checked-in NCU summaries")
    evidence.add_argument("--repo_root", type=Path, required=True)
    evidence.add_argument("--output", type=Path, required=True)
    return parser


def _workload_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_tokens", type=int, default=768)
    parser.add_argument("--output_tokens", type=int, default=128)
    parser.add_argument(
        "--precision_policy",
        default="",
        help="comma-separated component=bits, e.g. decoder_qkv=4,decoder_o=4",
    )


def _policy(text: str) -> dict[str, int]:
    if not text:
        return {}
    result = {}
    for item in text.split(","):
        key, value = item.split("=", 1)
        result[key.strip()] = int(value)
    return result


def _workload(args: argparse.Namespace) -> Workload:
    if getattr(args, "workload", None):
        return Workload.load(args.workload)
    return build_pointllm_7b_workload(
        batch_size=args.batch_size,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        precision_policy=_policy(args.precision_policy),
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "preset":
        workload = _workload(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        workload.save(args.output)
        print(f"wrote {len(workload.operations)} operations to {args.output}")
        return 0
    if args.command == "extract-evidence":
        payload = extract_ncu_evidence(args.repo_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(args.output)
        return 0

    config = HardwareConfig.load(args.config)
    workload = _workload(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "simulate":
        result = ArchitectureSimulator(config).run(workload)
        result.save(args.output_dir / "simulation.json")
        write_rows_csv(args.output_dir / "summary.csv", [summary_row(result)])
        if args.plot:
            plot_simulation(result, args.output_dir / "simulation.png")
        print(json.dumps(summary_row(result), indent=2))
        return 0

    sweep_spec = SweepSpec.from_dict(json.loads(args.sweep.read_text(encoding="utf-8")))
    rows = run_sweep(config, workload, sweep_spec)
    pareto = latency_resource_pareto(rows)
    write_rows_csv(args.output_dir / "sweep.csv", rows)
    write_rows_csv(args.output_dir / "pareto.csv", pareto)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"designs": len(rows), "pareto_designs": pareto[:50]}, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.plot:
        plot_sweep(rows, pareto, args.output_dir / "sweep.png")
    print(json.dumps({"designs": len(rows), "pareto": len(pareto), "best": rows[0]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
