"""Visualise diode-method perf benchmark output.

Reads logs/perf_diode_methods.csv (method, avg_time_us, relative_speedup_vs_wdf_accurate)
and produces a matplotlib bar chart comparing each method's per-block time.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, "logs", "perf_diode_methods.csv")
    out_path = os.path.join(here, "logs", "perf_diode_methods.png")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run perf_diode_methods_test first.")
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = ["#9b59b6", "#2ecc71", "#3498db", "#e74c3c"]
    bars = ax.bar(df["method"], df["avg_time_us"],
                  color=colors[: len(df)], edgecolor="black", linewidth=0.5)

    # Reference WDF-Accurate row (first row) is the perf baseline.
    baseline = float(df.iloc[0]["avg_time_us"])

    for bar, time, speedup in zip(bars, df["avg_time_us"],
                                   df["relative_speedup_vs_wdf_accurate"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h,
                f"{time:.2f} us", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

        if abs(speedup - 1.0) < 1e-6:
            label = "baseline"
        elif speedup > 1.0:
            label = f"{speedup:.2f}x faster"
        else:
            label = f"{1.0 / speedup:.2f}x slower"
        ax.text(bar.get_x() + bar.get_width() / 2.0, h * 0.5,
                label, ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")

    ax.set_ylabel("Avg time per 512-sample block (us)")
    ax.set_title("Diode-bridge solver methods\n"
                 "(20,000 iterations, 1.5 V sine input)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.axhline(baseline, color="grey", linestyle=":", linewidth=1, alpha=0.6)

    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    main()
