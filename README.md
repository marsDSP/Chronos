# Chronos

Stereo delay plugin (VST3 / AU / Standalone) — **work in progress**.

The bulk of the interesting work lives in the header-only DSP and math layers under `source/dsp/`, which serve as a live prototype ground for my nascent DSP library:

**MarsCore** 
1. A header-only SIMD-first math primitive lib: [7/6] Padé approximants for sin/cos/tan/sinh/cosh/tanh, first-order ADAA waveshapers, bit-level log/exp, and branchless reciprocal refinement.
2. SIMD-optimized DSP building blocks: aligned buffer abstractions, interpolating delay lines, Schroeder allpass diffusion, Ornstein–Uhlenbeck wow/flutter, s-plane curve-fit filters, and a WDF diode-bridge ducker with a closed-form Wright-Omega solver.

All SIMD code is written against SSE2 intrinsics and abstracted through [SIMDe](https://github.com/simd-everywhere/simde) for portability.

## Plugin features

- Delay: time (5–5000 ms), feedback, low/high-cut on the feedback path, stereo crossfeed, mono collapse
- Post-delay reverb: Schroeder diffusion, predelay, decay, HF/LF damping, modulation
- Wow & flutter: cosine drift with OU noise perturbation, three-partial capstan flutter
- Ducker: wave-digital-filter antiparallel diode pair, sidechain-driven via block-rate gain ramp
- Crossfade bypass on all major paths

## Build

**Requirements**: CMake ≥ 3.23, Clang or AppleClang (C++23), Python 3, Boost headers, macOS 11+.

```sh
git clone --recurse-submodules https://github.com/marsDSP/Chronos.git
cd Chronos
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target Chronos_VST3 -j$(nproc)
```

The VST3 and AU bundles land in `build/Chronos_artefacts/<Config>/`. A post-build script copies the VST3 to `~/Desktop/vst test/` for quick host scanning.

**Optional targets**

| Target | Description |
|---|---|
| `Chronos_AU` | Audio Unit (macOS only) |
| `Chronos_Standalone` | Standalone app |
| `AudioPluginHost` | JUCE plugin host for debug (built by default) |
| `Run_AudioPluginHost_VST3` | Build + launch host pointed at VST3 |
| `simd_harness` / `perf_harness` | Correctness and throughput test suites |

To skip Tracy profiling: `-DTRACY_ENABLE=OFF -DTRACY_BUILD_VIEWER=OFF`.

Universal binary (x86_64 + arm64) is the default on macOS. Single-arch: `-DCMAKE_OSX_ARCHITECTURES=arm64`.

## Tests

```sh
cmake --build build --target simd_harness
cmake --build build --target perf_harness
```

Python visualisation scripts for each harness live alongside the test sources in `tests/`.

## Structure

```
source/dsp/math/        Padé fast math, SIMD config
source/dsp/engine/      Interpolating delay engine
source/dsp/diffusion/   Allpass diffusion, reverb
source/dsp/modulation/  Wow, flutter, OU noise
source/dsp/filter/      S-plane curve-fit HP/LP
source/dsp/dynamics/    WDF diode-bridge ducker
source/dsp/buffers/     Aligned SIMD buffer abstractions
docs/CHRONOS_DSP_BIBLE.md  Full derivations for everything above
```

## License

See `LICENSE`.