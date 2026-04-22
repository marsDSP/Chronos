#!/usr/bin/env python3
"""Chronos aligned-buffer layout visualisation.

Illustrates the memory layout swap introduced alongside the s-plane filter
port:

  OLD: std::vector<float> bufferL, bufferR
       \u2192 two separately-allocated heap blocks with implementation-defined
         alignment (usually 16 B minimum), no SIMD-width padding of the
         per-channel sample count, no explicit channel pointer table.

  NEW: AlignedSIMDBuffer<float> circularDelayBuffer
       \u2192 one contiguous allocation through xsimd::aligned_allocator,
         aligned to xsimd::default_arch::alignment() (16 B on SSE2 / NEON,
         32 B on AVX, 64 B on AVX-512), with per-channel sample count
         padded up to a whole SIMD batch, and a fixed-size
         array-of-channel-pointers table matching the view's layout.

Generates a pair of side-by-side diagrams and writes them to:
  tests/simd_harness/logs/aligned_buffer_layout.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")
OUT_PATH = os.path.join(LOG_DIR, "aligned_buffer_layout.png")

PASS_COLOR   = "#2b8a3e"
FAIL_COLOR   = "#c92a2a"
OLD_COLOR    = "#c92a2a"
NEW_COLOR    = "#1971c2"
MUTED        = "#495057"
HIGHLIGHT    = "#f59f00"
PADDING_COLOR = "#dee2e6"
CHANNEL_L    = "#4dabf7"
CHANNEL_R    = "#f06595"


def draw_sample_row(ax, origin, num_samples, width_per_sample,
                    y_height, color, edge="black", label_every=16):
    """Draw num_samples unit cells starting at origin; label every Nth."""
    x0, y0 = origin
    for i in range(num_samples):
        ax.add_patch(Rectangle((x0 + i * width_per_sample, y0),
                               width_per_sample, y_height,
                               facecolor=color, edgecolor=edge, linewidth=0.25))
        if (i % label_every) == 0:
            ax.text(x0 + i * width_per_sample + width_per_sample * 0.5,
                    y0 - 0.3, f"{i}",
                    ha="center", va="top", fontsize=7, color=MUTED)


def alignment_markers(ax, origin, num_samples, width_per_sample,
                      y_bot, y_top, alignment_samples, color):
    """Draw vertical dashed lines at every alignment_samples boundary."""
    x0, _ = origin
    for i in range(0, num_samples + 1, alignment_samples):
        ax.plot([x0 + i * width_per_sample, x0 + i * width_per_sample],
                [y_bot, y_top],
                color=color, linestyle="--", linewidth=0.8, alpha=0.6)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # Stylised buffer parameters for the diagram (chosen small for clarity
    # rather than the engine's actual 262144-sample kBufSize).
    display_samples  = 64
    logical_samples  = 58            # the "requested" count
    simd_batch_width = 8             # e.g. AVX float batch
    padded_samples   = ((logical_samples + simd_batch_width - 1)
                        // simd_batch_width) * simd_batch_width
    padding_count    = padded_samples - logical_samples
    padded_samples   = max(padded_samples, display_samples)  # for display

    width_per_sample = 0.2
    y_height_old     = 0.5
    y_height_new     = 0.5

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Chronos circular-buffer migration  \u2014  "
                 "std::vector<float> (OLD) vs AlignedSIMDBuffer<float> (NEW)",
                 fontsize=15, fontweight="bold", color=MUTED)

    # ================================================================= OLD
    ax_old = fig.add_subplot(2, 1, 1)
    ax_old.set_xlim(-1.5, display_samples * width_per_sample + 1.5)
    ax_old.set_ylim(-1.8, 5.0)
    ax_old.set_axis_off()

    ax_old.text(-1.4, 4.4,
                "OLD  \u2014  std::vector<float> bufferL, bufferR",
                fontsize=13, fontweight="bold", color=OLD_COLOR)
    ax_old.text(-1.4, 3.9,
                "two independent heap allocations  \u2022  malloc alignment only  "
                "\u2022  no per-channel SIMD padding  \u2022  no pointer table",
                fontsize=9, color=MUTED)

    # Channel L row.
    y_L_old = 2.4
    draw_sample_row(ax_old, (0, y_L_old), logical_samples, width_per_sample,
                    y_height_old, CHANNEL_L)
    ax_old.text(-0.3, y_L_old + y_height_old * 0.5, "bufferL",
                ha="right", va="center", fontsize=10, color=MUTED,
                fontweight="bold")

    # Channel R row (separate allocation, visually offset).
    y_R_old = 0.8
    draw_sample_row(ax_old, (0.7, y_R_old), logical_samples, width_per_sample,
                    y_height_old, CHANNEL_R)
    ax_old.text(-0.3, y_R_old + y_height_old * 0.5, "bufferR",
                ha="right", va="center", fontsize=10, color=MUTED,
                fontweight="bold")

    # Annotate: unknown alignment - indicate with a hatched "address"
    # boundary marker.
    alignment_markers(ax_old, (0, y_L_old), logical_samples, width_per_sample,
                      y_L_old - 0.2, y_L_old + y_height_old + 0.2,
                      alignment_samples=1, color="#dee2e6")
    ax_old.text(0, y_L_old + y_height_old + 0.35,
                "unaligned, unpadded, separate heap blocks",
                fontsize=8, color=OLD_COLOR, style="italic")

    # Problem callouts.
    ax_old.text(logical_samples * width_per_sample / 2, -0.9,
                "\u2022  end-of-channel SIMD tail reads may walk off page\n"
                "\u2022  two independent allocations \u2192 cache-unfriendly\n"
                "\u2022  engine grabs &vec[0] but alignment not guaranteed\n"
                "\u2022  no BufferView adapter \u2192 must use juce::dsp::AudioBlock",
                fontsize=9, color=OLD_COLOR, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=OLD_COLOR))

    # ================================================================= NEW
    ax_new = fig.add_subplot(2, 1, 2)
    ax_new.set_xlim(-1.5, display_samples * width_per_sample + 1.5)
    ax_new.set_ylim(-1.8, 5.0)
    ax_new.set_axis_off()

    ax_new.text(-1.4, 4.4,
                "NEW  \u2014  AlignedSIMDBuffer<float> circularDelayBuffer",
                fontsize=13, fontweight="bold", color=NEW_COLOR)
    ax_new.text(-1.4, 3.9,
                f"xsimd::aligned_allocator  \u2022  batch-aligned  "
                f"\u2022  per-channel padded up to {simd_batch_width}-sample "
                f"boundary  \u2022  array-of-pointers table",
                fontsize=9, color=MUTED)

    # L channel: logical samples + SIMD padding
    y_L_new = 2.4
    draw_sample_row(ax_new, (0, y_L_new), logical_samples, width_per_sample,
                    y_height_new, CHANNEL_L)
    # Padding tail
    draw_sample_row(ax_new, (logical_samples * width_per_sample, y_L_new),
                    padding_count, width_per_sample, y_height_new,
                    PADDING_COLOR)

    # R channel (contiguous, right after L channel's padded end)
    y_R_new = 0.8
    draw_sample_row(ax_new, (0, y_R_new), logical_samples, width_per_sample,
                    y_height_new, CHANNEL_R)
    draw_sample_row(ax_new, (logical_samples * width_per_sample, y_R_new),
                    padding_count, width_per_sample, y_height_new,
                    PADDING_COLOR)

    ax_new.text(-0.3, y_L_new + y_height_new * 0.5, "bufferL",
                ha="right", va="center", fontsize=10, color=MUTED,
                fontweight="bold")
    ax_new.text(-0.3, y_R_new + y_height_new * 0.5, "bufferR",
                ha="right", va="center", fontsize=10, color=MUTED,
                fontweight="bold")

    # Alignment boundary markers (every SIMD batch).
    for y in (y_L_new, y_R_new):
        alignment_markers(ax_new, (0, y),
                          padded_samples, width_per_sample,
                          y, y + y_height_new + 0.05,
                          alignment_samples=simd_batch_width,
                          color=NEW_COLOR)

    # Label the padding region.
    if padding_count > 0:
        pad_x = (logical_samples + padding_count * 0.5) * width_per_sample
        ax_new.text(pad_x, y_L_new + y_height_new + 0.35,
                    "SIMD pad", ha="center", fontsize=8, color=MUTED,
                    style="italic")

    # Annotate requested vs padded sample counts.
    ax_new.annotate("",
                    xy=(0, y_L_new - 0.25),
                    xytext=(logical_samples * width_per_sample, y_L_new - 0.25),
                    arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.0))
    ax_new.text(logical_samples * width_per_sample / 2, y_L_new - 0.45,
                f"getNumSamples() = {logical_samples}",
                ha="center", fontsize=8, color=MUTED)

    ax_new.annotate("",
                    xy=(0, y_R_new - 0.25),
                    xytext=(padded_samples * width_per_sample, y_R_new - 0.25),
                    arrowprops=dict(arrowstyle="<->", color=NEW_COLOR, lw=1.0))
    ax_new.text(padded_samples * width_per_sample / 2, y_R_new - 0.45,
                f"getPaddedNumSamplesPerChannel() = {padded_samples}",
                ha="center", fontsize=8, color=NEW_COLOR)

    # Channel-pointer table diagram to the right.
    box_x = display_samples * width_per_sample + 0.2
    ptr_box_w = 1.1
    ptr_box_h = 0.45
    for i, (label, y_ch) in enumerate((("perChannelBasePointers[0]", y_L_new),
                                       ("perChannelBasePointers[1]", y_R_new))):
        ax_new.add_patch(FancyBboxPatch(
            (box_x, y_ch + y_height_new * 0.5 - ptr_box_h * 0.5),
            ptr_box_w, ptr_box_h,
            boxstyle="round,pad=0.04",
            facecolor="white", edgecolor=NEW_COLOR, linewidth=1.2))
        ax_new.text(box_x + ptr_box_w * 0.5,
                    y_ch + y_height_new * 0.5,
                    label, ha="center", va="center", fontsize=7, color=NEW_COLOR)
        # Arrow from pointer to channel start.
        ax_new.add_patch(FancyArrowPatch(
            (box_x, y_ch + y_height_new * 0.5),
            (0, y_ch + y_height_new * 0.5),
            arrowstyle="->",
            mutation_scale=12,
            color=NEW_COLOR, lw=1.0))

    # Benefit callouts.
    ax_new.text(logical_samples * width_per_sample / 2, -0.9,
                "\u2022  every channel base pointer satisfies SIMD alignment\n"
                "\u2022  tail reads within the padded region never fault\n"
                "\u2022  single contiguous allocation \u2192 cache-friendly\n"
                "\u2022  AlignedSIMDBufferView provides the process() API "
                "(no juce::dsp::AudioBlock needed)",
                fontsize=9, color=NEW_COLOR, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=NEW_COLOR))

    # Legend (shared across subplots).
    legend_patches = [
        mpatches.Patch(color=CHANNEL_L, label="channel 0 (L)"),
        mpatches.Patch(color=CHANNEL_R, label="channel 1 (R)"),
        mpatches.Patch(color=PADDING_COLOR, label="SIMD pad (zeros)"),
        plt.Line2D([], [], color=NEW_COLOR, linestyle="--",
                   label=f"{simd_batch_width}-sample alignment boundary"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, frameon=False)

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(OUT_PATH, dpi=130)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
