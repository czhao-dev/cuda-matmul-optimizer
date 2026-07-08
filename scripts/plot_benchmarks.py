#!/usr/bin/env python3
"""Generate benchmark PNGs from benchmarks/*.csv into benchmarks/plots/."""
import csv
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"
PLOTS = BENCH / "plots"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed hue-per-kernel identity, shared across every chart.
KERNEL_COLOR = {
    "cpu_baseline": INK_MUTED,
    "naive": "#2a78d6",
    "tiled": "#1baf7a",
    "vectorized": "#eda100",
    "coarsened": "#008300",
    "cublas_reference": "#4a3aa7",
}
KERNEL_LABEL = {
    "cpu_baseline": "CPU baseline",
    "naive": "Naive",
    "tiled": "Tiled",
    "vectorized": "Vectorized",
    "coarsened": "Coarsened",
    "cublas_reference": "cuBLAS",
}
KERNEL_ORDER = ["cpu_baseline", "naive", "tiled", "vectorized", "coarsened", "cublas_reference"]


def style_axes(ax, title):
    fig = ax.figure
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, pad=12)


def save(fig, name):
    PLOTS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS / name, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)


def read_csv(path, floats=(), ints=()):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in floats:
            row[key] = float(row[key])
        for key in ints:
            row[key] = int(row[key])
        if "kernel" in row:
            row["kernel"] = row["kernel"].removeprefix("gpu_")
    return rows


def plot_cuda_gflops_by_size(rows):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sizes = sorted({r["size_m"] for r in rows})
    for kernel in KERNEL_ORDER:
        pts = sorted((r["size_m"], r["gflops"]) for r in rows if r["kernel"] == kernel)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=2,
                color=KERNEL_COLOR[kernel], label=KERNEL_LABEL[kernel])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}×{s}" for s in sizes])
    ax.set_xlabel("Matrix size")
    ax.set_ylabel("GFLOP/s (log scale)")
    style_axes(ax, "CUDA Kernel Throughput by Matrix Size")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK_SECONDARY)
    save(fig, "cuda_gflops_by_size.png")


def plot_cuda_speedup_vs_cpu(rows):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    kernels = [k for k in KERNEL_ORDER if k != "cpu_baseline"]
    values = [next(r["speedup_vs_cpu"] for r in rows if r["kernel"] == k and r["size_m"] == 4096)
              for k in kernels]
    colors = [KERNEL_COLOR[k] for k in kernels]
    bars = ax.bar([KERNEL_LABEL[k] for k in kernels], values, color=colors, width=0.6, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("Speedup vs CPU baseline (log scale)")
    style_axes(ax, "Speedup vs CPU Baseline at 4096×4096")
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.0f}×", (bar.get_x() + bar.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=8.5, color=INK_PRIMARY)
    save(fig, "cuda_speedup_vs_cpu.png")


def main():
    cuda_rows = read_csv(BENCH / "results.csv", floats=("time_ms", "gflops", "speedup_vs_cpu"),
                          ints=("size_m",))
    plot_cuda_gflops_by_size(cuda_rows)
    plot_cuda_speedup_vs_cpu(cuda_rows)

    print(f"Wrote 2 PNGs to {PLOTS}")


if __name__ == "__main__":
    main()
