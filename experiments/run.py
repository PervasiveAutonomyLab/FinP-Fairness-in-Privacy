"""Run all experiments for a dataset and tee each run's output to a log file.

Examples
--------
List the experiments that would run for FEMNIST::

    python -m experiments.run --dataset femnist --list

Run the full FEMNIST matrix (baseline, FinP beta 0.5/0.75/1, DP noise 0.75/1/2),
then print + save a comparison table::

    python -m experiments.run --dataset femnist

Run only a subset, or do a quick smoke test with fewer rounds::

    python -m experiments.run --dataset femnist --only baseline,finp_beta1
    python -m experiments.run --dataset femnist --epochs 2

Just compare existing logs without running anything::

    python -m experiments.run --dataset femnist --compare-only
"""

import argparse
import datetime as _dt
import os
import subprocess
import sys

from experiments.config import EXPERIMENTS, DATASETS, Experiment
from experiments import compare as _compare

# Repo root = parent of this file's directory; main_fed.py lives there.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MAIN = os.path.join(_REPO_ROOT, "main_fed.py")


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _select(dataset: str, only: str):
    exps = EXPERIMENTS[dataset]
    if not only:
        return exps
    wanted = {n.strip() for n in only.split(",") if n.strip()}
    chosen = [e for e in exps if e.name in wanted]
    missing = wanted - {e.name for e in exps}
    if missing:
        raise SystemExit(
            f"Unknown experiment name(s) for '{dataset}': {sorted(missing)}.\n"
            f"Available: {[e.name for e in exps]}"
        )
    return chosen


def _build_command(exp: Experiment, epochs_override) -> list:
    args = list(exp.args)
    if epochs_override is not None:
        # Replace an existing --epochs N pair if present, else append.
        if "--epochs" in args:
            i = args.index("--epochs")
            args[i + 1] = str(epochs_override)
        else:
            args += ["--epochs", str(epochs_override)]
    return [sys.executable, _MAIN, *args]


def _run_one(cmd: list, log_path: str) -> int:
    """Run cmd from the repo root, streaming output to stdout and log_path (tee)."""
    print(f"\n$ {' '.join(cmd)}")
    print(f"  -> logging to {log_path}")
    with open(log_path, "w") as log:
        log.write("[Command] " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, cwd=_REPO_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
    return proc.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=DATASETS,
                        help="Which dataset's experiment matrix to run")
    parser.add_argument("--only", default="",
                        help="Comma-separated experiment names to run (default: all)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override --epochs for every run (handy for quick smoke tests)")
    parser.add_argument("--logdir", default=None,
                        help="Directory for logs (default: experiments/logs/<dataset>)")
    parser.add_argument("--list", action="store_true", help="List experiments and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands without running them")
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip running; just compare existing logs in --logdir")
    parser.add_argument("--no-compare", action="store_true",
                        help="Run experiments but skip the comparison table")
    args = parser.parse_args(argv)

    exps = _select(args.dataset, args.only)
    logdir = args.logdir or os.path.join(_REPO_ROOT, "experiments", "logs", args.dataset)
    os.makedirs(logdir, exist_ok=True)

    if args.list:
        print(f"Experiments for '{args.dataset}':")
        for e in exps:
            print(f"  {e.name:<22} python main_fed.py {' '.join(e.args)}")
        return 0

    if args.dry_run:
        for e in exps:
            print(" ".join(_build_command(e, args.epochs)))
        return 0

    labeled_logs = []
    if not args.compare_only:
        failures = []
        for e in exps:
            log_path = os.path.join(logdir, f"{e.name}_{_timestamp()}.log")
            rc = _run_one(_build_command(e, args.epochs), log_path)
            labeled_logs.append((e.name, log_path))
            if rc != 0:
                failures.append((e.name, rc))
                print(f"[run] WARNING: '{e.name}' exited with code {rc}", file=sys.stderr)
        if failures:
            print(f"[run] {len(failures)} run(s) failed: {failures}", file=sys.stderr)

    if args.no_compare:
        return 0

    # For compare-only, or to include any pre-existing logs, scan the directory
    # and keep the most recent log per experiment name.
    if args.compare_only or not labeled_logs:
        import glob, re
        latest = {}
        for f in sorted(glob.glob(os.path.join(logdir, "*.log"))):
            name = re.sub(r"_\d{8}_\d{6}$", "", os.path.splitext(os.path.basename(f))[0])
            latest[name] = f  # sorted ascending -> last wins (most recent timestamp)
        order = [e.name for e in exps]
        labeled_logs = [(n, latest[n]) for n in order if n in latest]
        labeled_logs += [(n, p) for n, p in latest.items() if n not in order]

    rows = _compare.build_rows(labeled_logs)
    if rows:
        print("\n" + "=" * 72)
        print(f"  EXPERIMENT COMPARISON for dataset '{args.dataset}'")
        print("  (attack metrics use MIA when --mia, else SIA)")
        print("=" * 72)
        print(_compare.render_table(rows))
        print("=" * 72)
        csv_path = os.path.join(logdir, f"comparison_{args.dataset}.csv")
        _compare.write_csv(rows, csv_path)
        print(f"\nComparison written to {csv_path}\n")
    else:
        print("[run] No RUN SUMMARY blocks found to compare.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
