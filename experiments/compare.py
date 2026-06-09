"""Parse the ``RUN SUMMARY`` block emitted by ``main_fed.py`` and tabulate runs.

Each ``main_fed.py`` run prints a ``RUN SUMMARY`` block at the end. This module
extracts the headline metrics requested for cross-experiment comparison:

    (1) mean of last 3 rounds train / test accuracy
    (2) mean of average attack accuracy
    (3) max  of average attack accuracy
    (4) mean average_loss_mad
    (5) mean Loss CoV and mean Loss FI
    (6) mean attack CoV and mean attack FI
    (7) mean sen_welfare

For runs with ``--mia`` the attack-based metrics (2)(3)(6)(7) are reported using
the MIA numbers instead of SIA, as requested. Loss-based metrics (4)(5) and
accuracy (1) are always the same series.

Usage:
    python -m experiments.compare path/to/logdir
    python -m experiments.compare run1.log run2.log --csv out.csv
"""

import argparse
import csv
import glob
import os
import re
import sys
from typing import Dict, List, Optional

_NUM = r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?|nan)"


def _last(text: str, pattern: str) -> Optional[float]:
    """Return the float from the LAST match of ``pattern`` (group 1), or None."""
    matches = re.findall(pattern, text)
    if not matches:
        return None
    val = matches[-1]
    try:
        return float(val)
    except ValueError:
        return None


def parse_summary(text: str) -> Optional[Dict[str, Optional[float]]]:
    """Parse the last RUN SUMMARY block in ``text`` into a flat metric dict.

    Returns None if no RUN SUMMARY block is present.
    """
    if "RUN SUMMARY" not in text:
        return None
    # Restrict to the last summary block so re-run logs report the final result.
    text = text[text.rindex("RUN SUMMARY"):]

    has_mia = "MIA attack accuracy" in text

    metrics: Dict[str, Optional[float]] = {
        "has_mia": 1.0 if has_mia else 0.0,
        # (1) accuracy
        "train_last3": _last(text, r"train %:\s*" + _NUM),
        "test_last3": _last(text, r"test %:\s*" + _NUM),
        # SIA attack metrics
        "sia_acc_mean": _last(text, r"mean of average SIA attack accuracy \(%\):\s*" + _NUM),
        "sia_acc_max": _last(text, r"max of average SIA attack accuracy \(%\):\s*" + _NUM),
        "sia_cov_mean": _last(text, r"mean Sia CoV:\s*" + _NUM),
        "sia_fi_mean": _last(text, r"mean Sia FI:\s*" + _NUM),
        "sia_sen_welfare_mean": _last(text, r"mean sen_welfare:\s*" + _NUM),
        # loss-based metrics (always reported)
        "loss_mad_mean": _last(text, r"mean average_loss_mad:\s*" + _NUM),
        "loss_cov_mean": _last(text, r"mean Loss CoV:\s*" + _NUM),
        "loss_fi_mean": _last(text, r"mean Loss FI:\s*" + _NUM),
        # MIA attack metrics (present only with --mia)
        "mia_acc_mean": _last(text, r"mean MIA attack accuracy:\s*" + _NUM),
        "mia_acc_max": _last(text, r"max MIA attack accuracy:\s*" + _NUM),
        "mia_cov_mean": _last(text, r"mean MIA CoV:\s*" + _NUM),
        "mia_fi_mean": _last(text, r"mean MIA FI:\s*" + _NUM),
        "mia_sen_welfare_mean": _last(text, r"mean reverse_mia sen_welfare \(across rounds\):\s*" + _NUM),
    }
    return metrics


# Final comparison columns. Each row resolves attack metrics to MIA when the run
# has --mia, otherwise SIA (see _row_from_metrics).
_COLUMNS = [
    ("experiment", "experiment"),
    ("attack", "attack"),
    ("train_last3", "train_last3(%)"),
    ("test_last3", "test_last3(%)"),
    ("attack_acc_mean", "attack_acc_mean"),
    ("attack_acc_max", "attack_acc_max"),
    ("attack_cov_mean", "attack_cov_mean"),
    ("attack_fi_mean", "attack_fi_mean"),
    ("loss_cov_mean", "loss_cov_mean"),
    ("loss_fi_mean", "loss_fi_mean"),
    ("loss_mad_mean", "loss_mad_mean"),
    ("sen_welfare_mean", "sen_welfare_mean"),
]


def _row_from_metrics(label: str, m: Dict[str, Optional[float]]) -> Dict[str, object]:
    """Collapse a parsed metric dict into the comparison row (MIA or SIA)."""
    use_mia = bool(m.get("has_mia"))
    if use_mia:
        return {
            "experiment": label,
            "attack": "MIA",
            "train_last3": m["train_last3"],
            "test_last3": m["test_last3"],
            "attack_acc_mean": m["mia_acc_mean"],
            "attack_acc_max": m["mia_acc_max"],
            "loss_mad_mean": m["loss_mad_mean"],
            "loss_cov_mean": m["loss_cov_mean"],
            "loss_fi_mean": m["loss_fi_mean"],
            "attack_cov_mean": m["mia_cov_mean"],
            "attack_fi_mean": m["mia_fi_mean"],
            "sen_welfare_mean": m["mia_sen_welfare_mean"],
        }
    return {
        "experiment": label,
        "attack": "SIA",
        "train_last3": m["train_last3"],
        "test_last3": m["test_last3"],
        "attack_acc_mean": m["sia_acc_mean"],
        "attack_acc_max": m["sia_acc_max"],
        "loss_mad_mean": m["loss_mad_mean"],
        "loss_cov_mean": m["loss_cov_mean"],
        "loss_fi_mean": m["loss_fi_mean"],
        "attack_cov_mean": m["sia_cov_mean"],
        "attack_fi_mean": m["sia_fi_mean"],
        "sen_welfare_mean": m["sia_sen_welfare_mean"],
    }


def _fmt(v: object) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.4f}"
    return str(v)


def build_rows(labeled_logs: List[tuple]) -> List[Dict[str, object]]:
    """labeled_logs: list of (label, file_path). Returns comparison rows."""
    rows = []
    for label, path in labeled_logs:
        try:
            with open(path, "r", errors="replace") as f:
                text = f.read()
        except OSError as e:
            print(f"[compare] WARNING: cannot read {path}: {e}", file=sys.stderr)
            continue
        m = parse_summary(text)
        if m is None:
            print(f"[compare] WARNING: no RUN SUMMARY in {path}", file=sys.stderr)
            continue
        rows.append(_row_from_metrics(label, m))
    return rows


def render_table(rows: List[Dict[str, object]]) -> str:
    headers = [h for _, h in _COLUMNS]
    keys = [k for k, _ in _COLUMNS]
    table = [headers] + [[_fmt(r.get(k)) for k in keys] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    lines = []
    for ri, row in enumerate(table):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if ri == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    return "\n".join(lines)


def write_csv(rows: List[Dict[str, object]], path: str) -> None:
    keys = [k for k, _ in _COLUMNS]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([h for _, h in _COLUMNS])
        for r in rows:
            w.writerow([("" if r.get(k) is None else r.get(k)) for k in keys])


def _label_from_filename(path: str) -> str:
    """Strip a trailing _YYYYmmdd_HHMMSS timestamp and .log from a filename."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"_\d{8}_\d{6}$", "", stem)


def _collect_inputs(paths: List[str]) -> List[tuple]:
    """Expand dirs to *.log and return [(label, path)] sorted by name."""
    labeled = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(glob.glob(os.path.join(p, "*.log"))):
                labeled.append((_label_from_filename(f), f))
        else:
            labeled.append((_label_from_filename(p), p))
    return labeled


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare RUN SUMMARY metrics across runs.")
    parser.add_argument("paths", nargs="+", help="Log files and/or directories of .log files")
    parser.add_argument("--csv", default=None, help="Optional path to write the comparison as CSV")
    args = parser.parse_args(argv)

    labeled = _collect_inputs(args.paths)
    rows = build_rows(labeled)
    if not rows:
        print("No RUN SUMMARY blocks found.", file=sys.stderr)
        return 1

    print("\n" + "=" * 72)
    print("  EXPERIMENT COMPARISON (attack metrics use MIA when --mia, else SIA)")
    print("=" * 72)
    print(render_table(rows))
    print("=" * 72 + "\n")

    if args.csv:
        write_csv(rows, args.csv)
        print(f"Comparison written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
