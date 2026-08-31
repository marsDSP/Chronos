#!/usr/bin/env python3
"""
    Regenerate and regression-check source/presets/FactoryPresets.h.

    The canonical preset definitions live in this script. Run without
    arguments to check the committed header against the canonical data.
    Run with --write to regenerate the header. Run with --staging <dir>
    to import preset files saved by the plugin into the canonical table.

    Exits 0 when the committed header matches the canonical data.
    Exits non-zero on drift, so the script doubles as a regression check.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

# ── Canonical preset definitions ──────────────────────────────────────
# Each preset is (name, bank, [(paramID, value), ...]).
# A parameter absent from the list takes its default.

PRESETS: list[tuple[str, str, list[tuple[str, float]]]] = [
    ("Init", "Basics", []),
    ("Slapback", "Basics", [
        ("delayTime", 110.0), ("delayTimeR", 110.0), ("timeLink", 1.0),
        ("delaySync", 0.0), ("feedback", 0.12), ("dampHz", 6500.0),
        ("loopCutHz", 90.0), ("crossFeed", 0.0), ("hpfFreq", 60.0),
        ("lpfFreq", 12000.0), ("mix", 28.0),
    ]),
    ("Quarter Note", "Sync", [
        ("delaySync", 1.0), ("delayDivision", 11.0), ("feedback", 0.45),
        ("crossFeed", 0.15), ("dampHz", 8000.0), ("mix", 35.0),
    ]),
    ("Dotted Eighth", "Sync", [
        ("delaySync", 1.0), ("delayDivision", 10.0), ("feedback", 0.50),
        ("crossFeed", 0.25), ("dampHz", 7000.0), ("mix", 32.0),
    ]),
    ("BBD Wash", "Analog", [
        ("delayMode", 1.0), ("delayTime", 420.0), ("delayTimeR", 420.0),
        ("feedback", 0.62), ("dampHz", 3200.0), ("loopCutHz", 120.0),
        ("crossFeed", 0.40), ("loopDrive", 4.0), ("enableDiffuser", 1.0),
        ("diffusion", 0.70), ("diffuserSize", 0.60), ("diffModDepth", 0.50),
        ("diffModRateHz", 0.30), ("lpfFreq", 6500.0), ("mix", 45.0),
    ]),
    ("Tape Drift", "Analog", [
        ("delayMode", 1.0), ("timeLink", 0.0), ("delayTime", 300.0),
        ("delayTimeR", 305.0), ("feedback", 0.55), ("delayModDepth", 12.0),
        ("delayModRateHz", 0.28), ("dampHz", 4500.0), ("loopDrive", 6.0),
        ("loopSatOrder", 2.0), ("drive", 6.0), ("bits", 12.0), ("mix", 40.0),
    ]),
    ("Ping Pong", "Space", [
        ("timeLink", 0.0), ("delayTime", 250.0), ("delayTimeR", 375.0),
        ("crossFeed", 0.85), ("feedback", 0.50), ("dampHz", 9000.0),
        ("mix", 38.0),
    ]),
    ("Ambient Bloom", "Space", [
        ("delayTime", 700.0), ("delayTimeR", 700.0), ("feedback", 0.78),
        ("enableDiffuser", 1.0), ("diffusion", 0.90), ("diffuserSize", 0.85),
        ("diffModDepth", 0.80), ("diffModRateHz", 0.20), ("dampHz", 5000.0),
        ("loopCutHz", 150.0), ("hpfFreq", 120.0), ("lpfFreq", 9000.0),
        ("mix", 55.0),
    ]),
]

HEADER_PATH = Path(__file__).resolve().parents[2] / "source" / "presets" / "FactoryPresets.h"


def _camel(name: str) -> str:
    """Convert a preset name to a C++ identifier fragment."""
    out = ""
    for word in name.replace(".", "").split():
        out += word.capitalize()
    return out


def _fmt_val(v: float) -> str:
    """Format a float for C++ source."""
    if v == int(v):
        return f"{v:.1f}f"
    return f"{v}f"


def generate_header() -> str:
    """Generate the full FactoryPresets.h content from PRESETS."""
    lines = [
        "#pragma once",
        "",
        "#ifndef CHRONOS_FACTORY_PRESETS_H",
        "#define CHRONOS_FACTORY_PRESETS_H",
        "",
        "#include <cstddef>",
        "",
        "namespace MarsDSP::Presets {",
        "",
        "// One parameter override in a factory preset.",
        "// A parameter absent from the table takes its default.",
        "struct FactoryPresetPair {",
        "    const char* paramID;",
        "    float value;",
        "};",
        "",
        "// One compiled-in factory preset.",
        "struct FactoryPreset {",
        "    const char* name;",
        "    const char* bank;",
        "    const FactoryPresetPair* pairs;",
        "    int numPairs;",
        "};",
        "",
    ]

    for name, bank, pairs in PRESETS:
        ident = _camel(name)
        lines.append(f"// {bank} / {name}.")
        if not pairs:
            lines.append(f"inline constexpr FactoryPresetPair k{ident}Pairs[] = {{}};")
        else:
            lines.append(f"inline constexpr FactoryPresetPair k{ident}Pairs[] = {{")
            for pid, val in pairs:
                lines.append(f'    {{ "{pid}", {_fmt_val(val)} }},')
            lines.append("};")
        lines.append("")

    lines.append("// The compiled-in factory bank. Read-only, listed first, never written to disk.")
    lines.append("inline constexpr FactoryPreset kFactoryPresets[] = {")
    for name, bank, pairs in PRESETS:
        ident = _camel(name)
        n = len(pairs)
        lines.append(f'    {{ "{name}",{"":>{max(1, 16 - len(name))}}"{bank}",'
                      f'{"":>{max(1, 12 - len(bank))}}k{ident}Pairs,{"":>{max(1, 5 - len(str(n)))}}{n} }},')
    lines.append("};")
    lines.append("")
    lines.append("inline constexpr int kNumFactoryPresets =")
    lines.append("    static_cast<int>(std::size(kFactoryPresets));")
    lines.append("")
    lines.append("} // namespace MarsDSP::Presets")
    lines.append("")
    lines.append("#endif")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate and check FactoryPresets.h")
    parser.add_argument("--write", action="store_true", help="write the generated header to disk")
    parser.add_argument("--staging", type=str, default=None,
                        help="read preset files from this directory and adopt them")
    args = parser.parse_args()

    if args.staging:
        print(f"Reading preset files from {args.staging} ...")
        # The staging workflow reads .chronos files saved by the plugin.
        # Each file is a state tree XML with presetName/presetAuthor/presetCategory.
        # This import path is a manual authoring step; the owner edits the
        # canonical PRESETS list above after reviewing the staging files.
        staging = Path(args.staging)
        if not staging.is_dir():
            print(f"ERROR: staging directory not found: {staging}")
            return 1
        files = sorted(staging.glob("*.chronos"))
        print(f"Found {len(files)} preset file(s).")
        for f in files:
            print(f"  {f.name}")
        print("Review these files, then update the PRESETS list in this script")
        print("and run with --write to regenerate the header.")
        return 0

    generated = generate_header()

    if args.write:
        HEADER_PATH.write_text(generated)
        print(f"Wrote {HEADER_PATH} ({len(PRESETS)} presets)")
        return 0

    # Regression check: compare the generated content with the committed file.
    if not HEADER_PATH.exists():
        print(f"ERROR: header not found at {HEADER_PATH}")
        return 1

    committed = HEADER_PATH.read_text()

    # Normalize: strip trailing whitespace from each line.
    gen_lines = [l.rstrip() for l in generated.splitlines()]
    com_lines = [l.rstrip() for l in committed.splitlines()]

    if gen_lines == com_lines:
        print(f"FactoryPresets.h matches the canonical data ({len(PRESETS)} presets).")
        return 0

    print("MISMATCH: FactoryPresets.h does not match the canonical data.")
    print(f"  generated {len(gen_lines)} lines, committed {len(com_lines)} lines")
    max_lines = max(len(gen_lines), len(com_lines))
    diffs = 0
    for i in range(max_lines):
        g = gen_lines[i] if i < len(gen_lines) else "<missing>"
        c = com_lines[i] if i < len(com_lines) else "<missing>"
        if g != c:
            diffs += 1
            if diffs <= 10:
                print(f"  line {i + 1}:")
                print(f"    generated: {g}")
                print(f"    committed: {c}")
    if diffs > 10:
        print(f"  ... and {diffs - 10} more difference(s).")
    print("Run with --write to regenerate the header from the canonical data.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
