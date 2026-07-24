# AGENTS.md

Conventions for LLM agents working in this repository.

## Important Instructions for Agents

Follow these guidelines when working on this codebase:

1. **Think first, then read**: Before making changes, think through the problem and read relevant files in the codebase. Understand the existing code before proposing modifications.

2. **Verify plans before major changes**: Before making any major changes, check in with the user to verify the plan. Get confirmation before proceeding with significant modifications.

3. **Provide high-level explanations**: At every step, give a high-level explanation of what changes were made. Keep explanations concise and focused on the "what" and "why."

4. **Keep changes simple**: Make every task and code change as simple as possible. Avoid massive or complex changes. Every change should impact as little code as possible. Simplicity is paramount.

5. **Maintain architecture documentation**: Keep this documentation file updated to describe how the architecture of the app works. Update relevant sections when making architectural changes.

6. **Never speculate about unread code**: Never make claims about code you haven't opened. If the user references a specific file, you MUST read the file before answering. Investigate and read relevant files BEFORE answering questions about the codebase. Give grounded, hallucination-free answers based on actual file contents.

## Commits

No commit may include any co-author / agent attribution line. I am NOT here to promote harnesses. 
I am firmly pro-LLM but I do NOT support this shameless promotional tactic. Never add a `Co-Authored-By:` (or equivalent) trailer to a commit message, regardless of
scope or tool. This is non-negotiable and applies to every commit. This especially happens when prompted to wrangle git, make a commit, or open a PR. This is FORBIDDEN, and failure to adhere to this hard rule will be seen/interpreted as deliberate sabotage.

## Project overview

Chronos is a **Nonlinear Delay Engine** JUCE audio plugin (see `README.md`). It is still early, but no longer a bare template: `ChronosProcessor::processBlock` runs a per-sample signal chain of **delay → (dry + wet) → output gain → TPDF dither → quantization** to a target bit depth, driven by parameters owned by `ChronosParameters` over an `AudioProcessorValueTreeState`, and `createEditor()` returns a `GenericAudioProcessorEditor`. It targets **Standalone**, **VST3**, and (on macOS) **AU** formats. The intended header-only DSP will live under `source/dsp/` (currently empty; the current delay uses JUCE's `dsp::DelayLine`). C++23.

Identity (set in `CMakeLists.txt`): company `MarsDSP`, BUNDLE_ID `com.marsdsp.Chronos`, manufacturer code `MDSP`, plugin code `CHRO`, version `0.1.0`. Remote: `https://github.com/marsDSP/Chronos.git`, branch `main`.

## Project structure

- `source/` — plugin sources, globbed into the `Chronos` target via `file(GLOB_RECURSE ...)` for `*.{cpp,h,hpp}`.
  - `ChronosProcessor.{h,cpp}` — JUCE `AudioProcessor`. Stereo in/out via `BusesProperties`, mono/stereo layouts supported (the DSP loops over `totalNumInputChannels`, so mono works), no MIDI in/out, not a synth, tail length `0.0`. `processBlock` early-returns on bypass (the in-place buffer already holds the dry input), else per sample: `delayLine.setDelay` (smoothed delay in samples) → push dry / pop wet → write `dry + wet` → output gain (dB → linear via `Decibels::decibelsToGain`) → per-sample TPDF dither + quantization to the target bit depth, using two independent xorshift32 streams (one per channel, seeded from `std::random_device`). The delay is a `dsp::DelayLine<float>` (default linear interpolation) prepared in `prepareToPlay` with `setMaximumDelayInSamples` from `ChronosParameters::maxDelayTime` (5000 ms). `getBypassParameter()` is overridden to return the bypass param so hosts wire it natively. Parameters live in an `AudioProcessorValueTreeState` whose layout is built by `ChronosParameters::createParameterLayout()` and accessed via a `ChronosParameters` member: `gain` (float, −12…+12 dB), `bits` (int, 1…32), `delayTime` (float, 5…5000 ms), and `bypass` (bool). `isBusesLayoutSupported` is `protected`, matching the base class. `getStateInformation`/`setStateInformation` are still stubs; programs are the single default program.
  - `ChronosParameters.h` — owns the APVTS parameter IDs and `createParameterLayout()`, casts the `AudioParameterFloat`/`Int`/`Bool` pointers in its constructor (via a `castParameter` helper, `jassert`-checked), and exposes smoothed, audio-thread-safe accessors. `prepare(sr)` stores the sample rate and sets a ~20 ms ramp on the gain/bits/delay smoothers; `reset()` snaps them to the current stored values; `update()` aims them at the current knob positions (once per block); `smoothen()` advances all three (once per sample). Delay time is smoothed in samples via `msToSamples`. All raw-parameter reads are null-guarded. Exposes `getBypassParameter()` for the processor's override.
  - `ChronosEditor.{h,cpp}` — JUCE `AudioProcessorEditor` from the template. Note the processor does not instantiate it; `createEditor()` returns a `GenericAudioProcessorEditor` instead.
  - `math/Trigonometry.h` — `mmSin`, `mmCos`, `mmTan`: rational minimax approximations, each with a `float` and an `M128` overload, all evaluated as a Horner series in x² and sharing the FMA helper `mulAdd` (`simde_mm_fmadd_ps`). `mmSin` is [7/6] odd on [−π, π] (float32 max abs err 3.2e-07); `mmCos` is [6/6] even on [−π, π] (3.5e-07); `mmTan` is [7/6] odd on [−1.55, 1.55], fitted for **relative** error because tan diverges at ±π/2 — its `Q(x²)` carries a root at ±π/2 to reproduce the pole, so callers must range-reduce and never pass |x| ≥ π/2. Coefficient namespaces are `Minimax{Sin,Cos,Tan}Coeffs`, with `minimax{Sin,Cos,Tan}Approx` as the scalar kernels. Caveat: `mmTan` is rounding-limited rather than coefficient-limited near the pole — `Q` falls to 0.023 at x = 1.55, so float32 cancellation caps relative error at ~3.9e-06 there (~1.9e-07 for |x| ≤ 1.0) no matter how good the coefficients are. `simd/Config.h` — the SIMD abstraction layer, providing the `MM()` / `M128` macros over native SSE or SIMDe. Neither header is `#include`d by the plugin yet, so neither is compiled by the build.
  - `dsp/`, `gui/` — empty placeholder directories for future header-only DSP / GUI code.
  - `utils/{data,helpers,memory}/` — empty placeholder directories for future utilities.
- `tests/` — scaffolded but NOT wired into the build. Empty placeholder subdirectories: `benchmarks/`, `charts/`, `harnesses/{cd,perf,simd}/`, `logs/`. There is no `tests/CMakeLists.txt` and no test target. `Catch2` is vendored under `libs/` for future numeric verification once `source/dsp/` is populated.
- `libs/` — third-party code as git submodules (marsDSP forks): `JUCE`, `simde`, `Catch2`, `pluginval`. No KFR, gcem, tracy, or xsimd are vendored. Note: `CMakeLists.txt` still lists `libs/xsimd/include` in the `SharedCode` include dirs, but that path does not exist yet (no xsimd submodule).
- `scripts/bash/copy_vst3.sh` — post-build copy of `Chronos.vst3` into `~/Desktop/vst test`. `scripts/python/remez_{sin,cos,tan}.py` each re-derive one set of minimax coefficients in `source/math/Trigonometry.h` via a linearized Remez exchange, seeded from the corresponding Padé approximant, and exit non-zero if the derived values drift from the header — so they double as regression checks on those constants (need numpy; run as `./.venv/bin/python scripts/python/remez_cos.py`). Each prints its equioscillation report (ripple ratio ≈ 1.0 when converged), a pole-free check on `Q`, and float32 error with and without FMA. `remez_tan.py` additionally reports the implied pole location (must land on π/2, above the fit interval) and a float32 error breakdown by sub-range. Two gotchas if you edit them: the extrema scan needs a **noise floor**, since the tan relative-error curve sits at float64 cancellation noise (~2e-16) as x → 0 and will otherwise report thousands of bogus alternations; and the node list must be padded from a uniform grid when a Padé seed yields fewer than the 8 required alternation points. There is no `setup.sh` or `build.sh`.
- `assets/{fonts,png,svg}/` — empty placeholders, globbed via `juce_add_binary_data(AudioPluginData)` only when non-empty. Currently empty, so the `AudioPluginData` target is not created.
- `docs/{notebooks,papers}/` — empty placeholders.
- `cmake/`, `tooling/` — empty placeholders.

CMake shape: a `SharedCode` **INTERFACE** library exposes the include paths (`source`, `libs`, `libs/JUCE/modules`, `libs/xsimd/include`, `libs/simde`, generated `JuceLibraryCode`) and `cxx_std_23`. The `Chronos` plugin target globs `source/**/*.{cpp,h,hpp}` and links `SharedCode` plus the JUCE modules (`juce_audio_basics`, `juce_audio_devices`, `juce_audio_formats`, `juce_audio_plugin_client`, `juce_audio_processors`, `juce_audio_utils`, `juce_core`, `juce_data_structures`, `juce_dsp`, `juce_events`, `juce_graphics`, `juce_gui_basics`, `juce_gui_extra`) and the JUCE recommended config/LTO/warning flags. `juce_generate_juce_header` produces `JuceHeader.h`. Python 3 is required (`find_package(Python3 COMPONENTS Interpreter REQUIRED)`); Boost and Doxygen are NOT required.

## Build

Chronos builds with CMake + JUCE. C++23, CMake ≥ 3.23.1. The primary workflow is CLion (`cmake-build-debug/`); standard CMake also works.

### Quick start

```sh
git clone --recurse-submodules https://github.com/marsDSP/Chronos.git
cd Chronos
cmake -S . -B build
cmake --build build
```

If you cloned without submodules:

```sh
git submodule update --init --recursive
```

### Prerequisites

- A C++23 toolchain (Xcode Command Line Tools on macOS).
- CMake ≥ 3.23.1 (Ninja optional; CLion ships its own generator).
- Python 3 (required by `CMakeLists.txt`).

### Targets

- `Chronos_Standalone`, `Chronos_VST3`, `Chronos_AU` (macOS only), `Chronos_All` — the plugin formats.
- `AudioPluginHost` — built from `libs/JUCE/extras/AudioPluginHost` (enabled via `BUILD_AUDIO_PLUGIN_HOST=ON`, the default).
- `Run_AudioPluginHost_VST3` / `Run_AudioPluginHost_AU` (macOS) — build the plugin + host, then open `Audio Plugin Host.app` for scanning.
- `Print_Plugin_Paths` — prints the VST3/AU artifact locations.

Built plugins are written to `<build>/Chronos_artefacts/<Config>/` as **Standalone**, **VST3**, and (on macOS) **AU** (e.g. `cmake-build-debug/Chronos_artefacts/Debug/VST3/Chronos.vst3`).

### macOS / build gotchas (read before touching CMake)

- **Universal binary**: `CMAKE_OSX_ARCHITECTURES` is forced to `x86_64;arm64`; deployment target `11.0` (including the arm64 slice).
- **x86_64 FMA**: the plugin target is compiled with `-Xarch_x86_64 -mfma` so `simde_mm_fmadd_ps` lowers to `vfmadd*ps` on the Intel slice (arm64 has FMA unconditionally and needs no flag). The `-Xarch_` prefix is required, since a bare `-mfma` would also be handed to the arm64 pass. This raises the x86_64 baseline to Haswell/FMA3 and switches that slice to VEX encoding; the 2013 Mac Pro (Ivy Bridge-EP) is still a supported macOS 11 machine and has AVX but no FMA3, so drop the flag if that hardware must keep working. Without it the code still builds — SIMDe falls back to a mul+add pair.
- **Warnings**: `-w` disables all warnings on non-MSVC compilers; JUCE recommended warning/LTO flags are still applied to the plugin target.
- **Debug symbols**: `-gdwarf-4` is added in `Debug`.
- **ObjC ARC**: on Apple, JUCE `.mm` sources and the plugin target are compiled with `-fno-objc-arc`.
- **Policy overrides**: `CMP0177` (install `DESTINATION` normalization) and `CMP0167` (Boost) are set to `NEW`. `CMAKE_POLICY_VERSION_MINIMUM` is not set, since no vendored lib currently needs it.
- **Submodule ordering**: `add_subdirectory(libs/JUCE)` runs at the top of `CMakeLists.txt`, before the macOS arch/deployment-target cache variables are set. This configures fine today, but if you add a vendored lib that reads `CMAKE_OSX_ARCHITECTURES` / `CMAKE_OSX_DEPLOYMENT_TARGET` at configure time, move those cache variables above its `add_subdirectory()`.
- **Post-build copy**: a `POST_BUILD` command on `Chronos_VST3` runs `scripts/bash/copy_vst3.sh`. The script currently ignores its `$1` argument and hardcodes `SOURCE_DIR` to `~/CLionProjects/Chronos/cmake-build-debug/Chronos_artefacts/Debug/VST3`, so it only works for that exact CLion Debug path — update it if you build elsewhere.
- **JUCE submodule**: `libs/JUCE` is a marsDSP fork. If it is empty after clone, run `git submodule update --init --recursive`.

## Testing

Not yet implemented. `tests/` contains only empty placeholder subdirectories (`benchmarks/`, `charts/`, `harnesses/{cd,perf,simd}/`, `logs/`) and there is no `tests/CMakeLists.txt`, so tests are not part of the CMake build. `Catch2` is vendored under `libs/` for future numeric verification of the DSP core once `source/dsp/` is populated. Do not assume any test target exists.

## Doxygen docs

No Doxygen target is defined. `docs/{notebooks,papers}/` are empty placeholders.
