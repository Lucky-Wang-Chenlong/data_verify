"""Export a run's metrics to plain files, or compare several runs.

    python tools/export_metrics.py run OUT_DIR
        OUT_DIR/metrics/train.csv   loss terms + LR per logged step
        OUT_DIR/metrics/val.csv     validation metrics per epoch
        OUT_DIR/results/summary.{json,md}  final AP of every eval under
                                    OUT_DIR/results/<mode>/ (from train.sh)
      For an eval-only directory (eval.sh) it writes OUT_DIR/results.json.

    python tools/export_metrics.py compare RUN_DIR... [--to FILE.md]
        One table: rows = runs, columns = AP per test set and prompt mode.

Scalars come from the tensorboard event files vis4d writes into the run
directory; eval numbers come from the "3D/<metric>: <value>" log lines.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

METRIC_LINE = re.compile(r"\] ?Vis4D[^:]*: (3D/[\w./-]+): (-?[\d.]+)")


def read_scalars(run_dir: str) -> dict[str, list[tuple[int, float]]]:
    """All tensorboard scalars of a run, merged over its event files."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    scalars: dict[str, dict[int, float]] = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents*"))):
        acc = EventAccumulator(path, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            for event in acc.Scalars(tag):
                scalars[tag][event.step] = event.value
    return {tag: sorted(v.items()) for tag, v in scalars.items()}


def write_wide_csv(path: str, scalars: dict, tags: list[str], extra=None):
    """One row per step, one column per tag (blank where not logged)."""
    steps = sorted({s for t in tags for s, _ in scalars[t]})
    table = {t: dict(scalars[t]) for t in tags}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        cols = ["step"] + (["epoch"] if extra else []) + tags
        w.writerow(cols)
        for step in steps:
            row = [step] + ([extra.get(step, "")] if extra else [])
            row += [table[t].get(step, "") for t in tags]
            w.writerow(row)
    return len(steps)


def read_eval(run_dir: str) -> dict[str, float]:
    """Final "3D/..." metrics printed by the evaluator (last value wins)."""
    metrics: dict[str, float] = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "log_*.txt"))):
        with open(path, errors="replace") as f:
            for line in f:
                m = METRIC_LINE.search(line)
                if m:
                    metrics[m.group(1)] = float(m.group(2))
    return metrics


def headline(metrics: dict[str, float]) -> dict[str, float]:
    """Per-test-set AP (3D/AP_<set>), the numbers to compare."""
    return {k[len("3D/AP_"):]: v for k, v in metrics.items()
            if k.startswith("3D/AP_")}


def export_run(out: str) -> None:
    scalars = read_scalars(out)
    train_tags = sorted(t for t in scalars
                        if t.startswith(("train/", "lr-")))
    val_tags = sorted(t for t in scalars if t.startswith("val/"))
    if train_tags:
        n = write_wide_csv(os.path.join(out, "metrics", "train.csv"),
                           scalars, train_tags)
        print(f"metrics/train.csv: {n} steps x {len(train_tags)} columns")
    if val_tags:
        epoch = dict(scalars.get("epoch", []))
        n = write_wide_csv(os.path.join(out, "metrics", "val.csv"),
                           scalars, val_tags, extra=epoch)
        print(f"metrics/val.csv: {n} validations x {len(val_tags)} columns")

    modes = sorted(d for d in glob.glob(os.path.join(out, "results", "*"))
                   if os.path.isdir(d))
    if modes:
        summary = {os.path.basename(d): read_eval(d) for d in modes}
        with open(os.path.join(out, "results", "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        lines = ["| prompt | " + " | ".join(_sets(summary)) + " |",
                 "|---|" + "---|" * len(_sets(summary))]
        for mode, m in summary.items():
            ap = headline(m)
            lines.append(f"| {mode} | " + " | ".join(
                _fmt(ap.get(s)) for s in _sets(summary)) + " |")
        with open(os.path.join(out, "results", "summary.md"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("results/summary.md:\n" + "\n".join(lines))
    elif os.path.exists(os.path.join(out, "eval.log")):  # eval.sh directory
        metrics = read_eval(out)
        with open(os.path.join(out, "results.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print("results.json:", json.dumps(headline(metrics)))


def _sets(summary: dict) -> list[str]:
    return sorted({s for m in summary.values() for s in headline(m)})


def _fmt(v) -> str:
    return "–" if v is None else f"{v:.4f}"


def compare(runs: list[str], to: str | None) -> None:
    rows, sets = {}, set()
    for run in runs:
        name = os.path.basename(os.path.normpath(run))
        files = {"summary": os.path.join(run, "results", "summary.json"),
                 "single": os.path.join(run, "results.json")}
        if os.path.exists(files["summary"]):
            with open(files["summary"]) as f:
                per_mode = {k: headline(v) for k, v in json.load(f).items()}
        elif os.path.exists(files["single"]):
            with open(files["single"]) as f:
                mode = next((m for m in ("box_depth", "text_depth", "box", "text")
                             if name.endswith(m)), "-")
                per_mode = {mode: headline(json.load(f))}
        else:
            print(f"skip {run}: no results (run `export_metrics.py run` first)")
            continue
        for mode, ap in per_mode.items():
            rows[(name, mode)] = ap
            sets.update(ap)
    sets = sorted(sets)
    lines = ["| run | prompt | " + " | ".join(sets) + " |",
             "|---|---|" + "---|" * len(sets)]
    for (name, mode), ap in rows.items():
        lines.append(f"| {name} | {mode} | " +
                     " | ".join(_fmt(ap.get(s)) for s in sets) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    if to:
        with open(to, "w") as f:
            f.write(text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+")
    c.add_argument("--to")
    args = p.parse_args()
    if args.cmd == "run":
        export_run(args.out)
    else:
        compare(args.runs, args.to)


if __name__ == "__main__":
    main()
