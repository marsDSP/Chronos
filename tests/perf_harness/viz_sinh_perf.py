"""Visualise sinh perf benchmark output.

Reads logs/perf_sinh_results.csv (algorithm, avg_time_us, speedup) and produces
a matplotlib bar chart comparing each algorithm's mean per-block time, with the
speedup-vs-baseline annotated above each bar.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "logs", "perf_sinh_results.csv")
    out_path = os.path.join(script_dir, "logs", "perf_sinh.png")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run perf_sinh_test first (from project root).")
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    bars = ax.bar(df["algorithm"], df["avg_time_us"],
                  color=colors[: len(df)], edgecolor="black", linewidth=0.5)

    for bar, time, speedup in zip(bars, df["avg_time_us"], df["speedup"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h,
                f"{time:.3f} us", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
        label = "baseline" if abs(speedup - 1.0) < 1e-6 else f"{speedup:.2f}x faster"
        ax.text(bar.get_x() + bar.get_width() / 2.0, h * 0.5,
                label, ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")

    ax.set_ylabel("Avg time per 512-sample block (us)")
    ax.set_title("SIMD Pade sinh performance\n(1,000,000 iterations, x in [-pi, pi])")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    main()
