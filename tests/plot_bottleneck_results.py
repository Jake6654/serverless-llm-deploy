# tests/plot_bottleneck_results.py
# ---------------------------------------------------------
# Plotting utility for bottleneck_results.csv
# Generates PNG charts into /app/outputs (or --outdir).
# No pandas dependency; uses csv + matplotlib only.
# ---------------------------------------------------------

import os
import csv
import math
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            # cast numbers
            for k in ["steps", "width", "height"]:
                r[k] = int(r[k])
            for k in [
                "scale",
                "pre_attach", "attach", "gen", "detach", "post_detach",
                "total_per_iter",
            ]:
                r[k] = float(r[k])
            rows.append(r)
    return rows


def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def fig_save(fig, outdir: Path, name: str):
    out = outdir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"[saved] {out}")
    plt.close(fig)


def plot_all_cases_stacked(rows, outdir):
    # Sort by name for a stable order
    rows_sorted = sorted(rows, key=lambda r: r["name"])
    labels = [r["name"] for r in rows_sorted]
    attach = [r["attach"] for r in rows_sorted]
    gen = [r["gen"] for r in rows_sorted]
    detach = [r["detach"] for r in rows_sorted]

    x = range(len(rows_sorted))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    ax.bar(x, attach, label="attach")
    ax.bar(x, gen, bottom=attach, label="generate")
    ax.bar(
        x,
        detach,
        bottom=[a + g for a, g in zip(attach, gen)],
        label="detach",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("time (s)")
    ax.set_title("Per-case latency breakdown (attach + generate + detach)")
    ax.legend()
    fig_save(fig, outdir, "all_cases_stacked")
    

def plot_steps_sweep(rows, outdir):
    b_rows = [r for r in rows if r["name"].startswith("B_")]
    if not b_rows:
        return
    # group by steps (already one per steps in our default set)
    b_rows = sorted(b_rows, key=lambda r: r["steps"])
    steps = [r["steps"] for r in b_rows]
    gen = [r["gen"] for r in b_rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, gen, marker="o")
    ax.set_xlabel("steps")
    ax.set_ylabel("generation time (s)")
    ax.set_title("Step sweep (expected ~ linear growth)")
    for s, g in zip(steps, gen):
        ax.annotate(f"{g:.2f}s", (s, g), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
    fig_save(fig, outdir, "steps_sweep")


def plot_resolution_sweep(rows, outdir):
    c_rows = [r for r in rows if r["name"].startswith("C_")]
    if not c_rows:
        return
    # sort by area
    c_rows = sorted(c_rows, key=lambda r: r["width"] * r["height"])
    area = [r["width"] * r["height"] for r in c_rows]
    gen = [r["gen"] for r in c_rows]
    labels = [f"{r['width']}x{r['height']}" for r in c_rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(area, gen, marker="o")
    ax.set_xlabel("pixels (H×W)")
    ax.set_ylabel("generation time (s)")
    ax.set_title("Resolution sweep (expected ~ quadratic growth)")
    for a, g, lab in zip(area, gen, labels):
        ax.annotate(f"{lab}\n{g:.2f}s", (a, g), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    fig_save(fig, outdir, "resolution_sweep")


def plot_lora_overhead(rows, outdir):
    # Expect names: A_base_only, A_lora_cold, A_lora_cached
    keys = ["A_base_only", "A_lora_cold", "A_lora_cached"]
    data = []
    for k in keys:
        m = next((r for r in rows if r["name"] == k), None)
        if m:
            data.append(m)

    if len(data) < 2:
        return

    labels = [r["name"].replace("A_", "") for r in data]
    total = [r["total_per_iter"] for r in data]
    attach = [r["attach"] for r in data]
    gen = [r["gen"] for r in data]
    detach = [r["detach"] for r in data]

    x = range(len(data))
    width = 0.7

    fig, ax = plt.subplots(figsize=(7, 4))
    # Stacked to show breakdown; also annotate total
    ax.bar(x, attach, width, label="attach")
    ax.bar(x, gen, width, bottom=attach, label="generate")
    ax.bar(x, detach, width, bottom=[a + g for a, g in zip(attach, gen)], label="detach")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("time (s)")
    ax.set_title("LoRA overhead: base vs cold-attach vs cached")
    for i, t in enumerate(total):
        ax.annotate(f"{t:.2f}s", (i, t), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)

    ax.legend()
    fig_save(fig, outdir, "lora_overhead_breakdown")


def main():
    p = argparse.ArgumentParser(description="Plot charts from bottleneck_results.csv")
    p.add_argument("--csv", default="bottleneck_results.csv")
    p.add_argument("--outdir", default="outputs")
    args = p.parse_args()

    outdir = ensure_outdir(Path(args.outdir))
    rows = load_rows(args.csv)
    if not rows:
        print("No rows found in CSV. Did you run the benchmark first?")
        return

    plot_all_cases_stacked(rows, outdir)
    plot_steps_sweep(rows, outdir)
    plot_resolution_sweep(rows, outdir)
    plot_lora_overhead(rows, outdir)

    print("\n[done] charts generated.")


if __name__ == "__main__":
    main()
