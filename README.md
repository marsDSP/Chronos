# Chronos
Nonlinear Delay Engine.

A stereo delay effect with a recirculating feedback loop, an in-loop diffuser,
a tanh saturation stage (antiderivative-antialiased), Digital/Analog output
filters, and a Digital/BBD delay core.

## Formats & platforms

- **Formats**: Standalone, VST3; AU (macOS only).
- **macOS 11.0+** (universal `x86_64` + `arm64`). The x86_64 slice needs an
  **FMA3-capable CPU** (Intel Haswell, 2013, or newer) — FMA is compiled in,
  so older Intel Macs will crash on launch.
- **Windows 10+ x64** — needs an AVX2-capable CPU (`/arch:AVX2` baseline).
- **Linux x86_64** — needs ALSA/JACK dev headers to build; FMA3 CPU.
- Mono or stereo in/out. No MIDI input; not a synth.

## Build

```sh
git clone --recurse-submodules https://github.com/marsDSP/Chronos.git
cd Chronos
cmake -S . -B build
cmake --build build
```

Requires CMake ≥ 3.23.1, a C++23 toolchain, and Python 3. Artefacts land in
`build/Chronos_artefacts/<Config>/`.

## Install

- **VST3** — copy `Chronos.vst3` to:
  - macOS: `/Library/Audio/Plug-Ins/VST3` (or `~/Library/Audio/Plug-Ins/VST3`)
  - Windows: `C:\Program Files\Common Files\VST3`
  - Linux: `~/.vst3`
- **AU (macOS)** — copy `Chronos.component` to `/Library/Audio/Plug-Ins/Components`.
- **Standalone (macOS)** — run `Chronos.app` from anywhere.

## Presets

Preset files (`.chronos`) live under the user data directory:

- macOS: `~/Library/Application Support/MarsDSP/Chronos/Presets`
- Windows: `%APPDATA%\MarsDSP\Chronos\Presets`
- Linux: `~/.config/MarsDSP/Chronos/Presets`

One level of subdirectory is a bank. The factory bank is built in and is
never written to disk.

## Known issues (beta)

- Switching **Drive Sat** (the ADAA order) while audio runs can produce a
  short click — the saturation and latency-alignment kernel change together.
- When the host supplies no tempo, the engine holds the last known BPM
  (default 120).
- Sessions saved by versions older than schema 5 are migrated on load; the
  removed `interpolation` parameter is dropped.

## Bugs

Open an issue at <https://github.com/marsDSP/Chronos/issues> with your OS,
host DAW and version, plugin format, and steps to reproduce.
