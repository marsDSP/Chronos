# Chronos
Nonlinear Delay Engine.

![CI](https://github.com/marsDSP/Chronos/actions/workflows/ci.yml/badge.svg)

## Transport & latency invariants

1. **PDC latency is a compile-time constant** — `SaturatorAlign::kBudget` (8 samples), independent of feedback amount, diffuser state, and `adaaOrder`. Enforced by `latency_null_check`, `diffuser_toggle_check`, and `diffusion_onset_check`.

2. **The feedback loop period is exactly `d`** — the saturator latency is absorbed into the loop tap (read at `d − satLatency`), so repeats recirculate on the grid. Enforced by `fb_parity`.

3. **The diffuser base transport is absorbed at the tap position** (C7) — the 8-section Schroeder cascade carries a series delay of `Σ effᵢ(size)` that is *pure* delay at `diffusion = 0` (each section degenerates to `y = d`) and remains the center of energy at all `g`. The engine subtracts this transport from the non-feedback delay request, and from a separate output tap on the feedback path (the loop period is untouched). Measured base transport (48 kHz):

   | `size` | 0.0 | 0.5 | 1.0 |
   |--------|-----|-----|-----|
   | transport | 611 ms | 336 ms | 61 ms |

   The instantaneous feedthrough of the cascade is `h[0] = g⁸` with `g = 0.92·diffusion`:

   | `diffusion` | 0.3 | 0.7 (default) | 1.0 |
   |-------------|-----|-----|-----|
   | `h[0]` | −89.5 dB | −30.6 dB | −5.8 dB |

   So with compensation the smear blooms *around* the repeat (pre-arrivals land ahead of the grid at high diffusion — the desired symmetric bloom) instead of shifting every repeat hundreds of ms late. Compensation is per the **mean** of the L/R transports, which preserves the deliberate ~3.8 ms L−R skew (intentional decorrelation, decision D2). When `delay < Σ effᵢ(size)` the tap clamps (`kMinLoopDelay` on the feedback path, 0 on the plain tap) and repeats land late by the remainder — full compensation at size 0 requires delay > 611 ms (decision D1). Enforced by `diffusion_onset_check` (diffusion 0: onset aligned to ≤144 samples; 0.7: energy centroid ≤144 samples; latency unchanged).

4. **Chunked feedback invariant** — `FeedbackDelay` processes in sub-chunks of `Lc ≤ D − 6` samples (D = minimum read delay across the loop and output taps, 6 = interpolator window), so no read in a chunk touches a write from the same chunk. Same invariant as `Diffuser::chunk_` (`kMinDelay 32 > kChunk 16`). Enforced bit-exact by `fb_parity` (satOrder 0) and to bounded noise (1e-5 rel / ±0.1 dB envelope) at satOrder 1/2.
