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

Chronos is a **Nonlinear Delay Engine** JUCE audio plugin (see `README.md`). It is still early, but no longer a bare template: `ChronosProcessor::processBlock` applies an output-gain stage followed by per-sample TPDF dither and quantization to a target bit depth, driven by an `AudioProcessorValueTreeState`, and `createEditor()` returns a `GenericAudioProcessorEditor`. It targets **Standalone**, **VST3**, and (on macOS) **AU** formats. The intended header-only DSP will live under `source/dsp/` (currently empty). C++23.

Identity (set in `CMakeLists.txt`): company `MarsDSP`, BUNDLE_ID `com.marsdsp.Chronos`, manufacturer code `MDSP`, plugin code `CHRO`, version `0.1.0`. Remote: `https://github.com/marsDSP/Chronos.git`, branch `main`.

## Project structure

- `source/` — plugin sources, globbed into the `Chronos` target via `file(GLOB_RECURSE ...)` for `*.{cpp,h,hpp}`.
  - `ChronosProcessor.{h,cpp}` — JUCE `AudioProcessor`. Stereo in/out via `BusesProperties`, mono/stereo layouts supported, no MIDI in/out, not a synth, tail length `0.0`. `processBlock` applies output gain (dB → linear via `Decibels::decibelsToGain`) then per-sample TPDF dither and quantization to the target bit depth, using two independent xorshift32 streams (one per channel, seeded from `std::random_device`). Parameters live in an `AudioProcessorValueTreeState` built by `createParameterLayout()`: `gain` (float, −12…+12 dB) and `bits` (int, 1…32). `isBusesLayoutSupported` is `protected`, matching the base class. `getStateInformation`/`setStateInformation` are still stubs; programs are the single default program.
  - `ChronosEditor.{h,cpp}` — JUCE `AudioProcessorEditor` from the template. Note the processor does not instantiate it; `createEditor()` returns a `GenericAudioProcessorEditor` instead.
  - `math/Trigonometry.h` — `pSin`, a minimax [7/6] odd-rational sine (scalar and `M128` overloads). `simd/Config.h` — the SIMD abstraction layer, providing the `MM()` / `M128` macros over native SSE or SIMDe. Neither header is `#include`d by the plugin yet, so neither is compiled by the build.
  - `dsp/`, `gui/` — empty placeholder directories for future header-only DSP / GUI code.
  - `utils/{data,helpers,memory}/` — empty placeholder directories for future utilities.
- `tests/` — scaffolded but NOT wired into the build. Empty placeholder subdirectories: `benchmarks/`, `charts/`, `harnesses/{cd,perf,simd}/`, `logs/`. There is no `tests/CMakeLists.txt` and no test target. `Catch2` is vendored under `libs/` for future numeric verification once `source/dsp/` is populated.
- `libs/` — third-party code as git submodules (marsDSP forks): `JUCE`, `simde`, `Catch2`, `pluginval`. No KFR, gcem, tracy, or xsimd are vendored. Note: `CMakeLists.txt` still lists `libs/xsimd/include` in the `SharedCode` include dirs, but that path does not exist yet (no xsimd submodule).
- `scripts/bash/copy_vst3.sh` — post-build copy of `Chronos.vst3` into `~/Desktop/vst test`. `scripts/python/remez_sin.py` re-derives the minimax sine coefficients in `source/math/Trigonometry.h` via a Remez exchange and exits non-zero if they drift from the header, so it doubles as a regression check on those constants (needs numpy; run it as `./.venv/bin/python scripts/python/remez_sin.py`). There is no `setup.sh` or `build.sh`.
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
