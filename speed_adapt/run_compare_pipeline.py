from __future__ import annotations

"""
Run compare + plot pipeline in sequence.

Usage example:
  python speed_adapt/run_compare_pipeline.py > /tmp/liseco.log 2>&1
"""

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _alpha_tag(alpha: float) -> str:
    return f"a{float(alpha):.1f}"


def _missing_calibration_alphas(calibration_base_dir: Path, alphas: list[float]) -> list[float]:
    missing: list[float] = []
    for alpha in alphas:
        cal_path = calibration_base_dir / _alpha_tag(alpha) / "calibration.json"
        if not cal_path.exists():
            missing.append(float(alpha))
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alphas",
        nargs="*",
        type=float,
        default=[1.0],
        help="Alphas to use for calibration, compare, and plotting.",
    )
    parser.add_argument(
        "--skip-clean-plot",
        action="store_true",
        help="Skip the clean:plot variant.",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip auto-building missing calibration files.",
    )
    parser.add_argument(
        "--calibrate-always",
        action="store_true",
        help="Always run calibration for requested alphas before compare.",
    )
    parser.add_argument(
        "--calibration-base-dir",
        type=Path,
        default=Path("data/speed_adapt/debug_calibrate_speed_signals/full_run"),
        help="Directory containing a{alpha}/calibration.json files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    py = sys.executable
    alphas = [float(a) for a in args.alphas]
    alpha_args = [str(a) for a in alphas]

    # 0) Ensure required calibration files exist (unless skipped)
    if not args.skip_calibration:
        if args.calibrate_always:
            print(
                "[info] running calibration for requested alphas:",
                ", ".join(alpha_args),
                flush=True,
            )
            _run(
                [py, "speed_adapt/debug_calibrate_speed_signals.py", "--alphas", *alpha_args],
                cwd=repo_root,
            )
        else:
            missing = _missing_calibration_alphas(args.calibration_base_dir, alphas)
            if missing:
                missing_args = [str(a) for a in missing]
                print(
                    "[info] missing calibration files for alphas:",
                    ", ".join(missing_args),
                    flush=True,
                )
                _run(
                    [py, "speed_adapt/debug_calibrate_speed_signals.py", "--alphas", *missing_args],
                    cwd=repo_root,
                )
            else:
                print("[info] calibration files already present for requested alphas", flush=True)

    # 1) Main compare run
    _run(
        [py, "speed_adapt/compare_mean_steering_adaptive_schedule_instruct.py", "--alphas", *alpha_args],
        cwd=repo_root,
    )

    # 2) Standard plot refresh
    _run(
        [py, "speed_adapt/plot_schedule_fullrun_metrics.py", "--alphas", *alpha_args],
        cwd=repo_root,
    )

    # 3) clean:plot refresh
    if not args.skip_clean_plot:
        _run(
            [
                py,
                "speed_adapt/plot_schedule_fullrun_metrics.py",
                "--alphas",
                *alpha_args,
                "--clean-plot",
            ],
            cwd=repo_root,
        )

    print("[done] compare + plot pipeline completed", flush=True)


if __name__ == "__main__":
    main()
