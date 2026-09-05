# Chronos — memory-behavior audit notes (C11)

Recorded so these conclusions are not re-litigated. Audited at the state of
`perf(engine): fuse per-sample stages 5+6 and 9a into 9b's loop` (C10).

## 1. Cache-coherence: clean

The RT path is single-threaded by construction. What was checked:

- `source/` is free of `std::mutex`, locks, `shared_ptr`, and threads
  (grep-clean). Two atomics sit on the RT path (H1), both relaxed, both
  best-effort scalars that publish no pointer:
  1. `ChronosProcessor::cachedBpm_` (`std::atomic<double>`) — the audio
     thread stores the host BPM each block; `computeDelaySamples_` (audio)
     and `getTailLengthSeconds` (host) load it.
  2. `ChronosProcessor::editorOpen_` (`std::atomic<bool>`) — the message
     thread sets it in `createEditor`/`setEditorOpen`; the audio thread
     loads it to gate metering and the tap-frame FIFO push.
  The `SpscFifo` index atomics are the sanctioned lock-free audio→GUI
  handoff (256-deep, editor-gated). The remaining `std::atomic` hits
  (`ChronosEditor`'s pending flags, `SegmentButtons::pendingValue_`,
  `PresetManager::modified_`) are message-thread-only and never touched
  from `process()`.
- Parameter flow is one-directional: APVTS raw-parameter pointers →
  `ChronosParameters` block-rate getters (relaxed atomic loads owned by the
  JUCE parameter objects, in `parameters.update()` / `getRaw*()`) → plain-POD
  `ChronosEngine::Params` by value → engine. No audio-thread atomic stores.
- No shared mutable metering in the RT path (`utils/data/MeteringFrame.h` is
  an unused placeholder, referenced only by itself).
- All ring/scratch/arena storage is touched by the audio thread only; the
  arena is sized and carved entirely inside `prepare()`. No false-sharing
  candidates: no two writers share a cache line, because there is only one
  writer.

## 2. Memory map (48 kHz, maxBlock 512 → wetBufCapacity 1024)

Post-C1/C9/C9b, the whole DSP layer is ONE `BumpArena` allocation —
`arena_.get_total_num_bytes()` = **4,625,664 B (~4.41 MB)**:

| Region | Floats | Bytes | Notes |
|---|---|---|---|
| delay rings (2) | 2 × 262,160 | ~2.0 MB | cap 262,144 + tail, 64 B-padded |
| feedback rings (2) | 2 × 262,160 | ~2.0 MB | same cap (C1 halved these from 512 K-sample rings) |
| diffuser rings (16) | 90,368 | ~353 KB | prime-snapped sections 1,163–6,353 + headroom |
| scratch spans (17) | 17 × 1,024 | 68 KB | stride padded to 16-float multiples |

Pre-C9b these were 21 separate heap regions; now every ring and span is
64-byte aligned inside one extent. At other block sizes only the scratch
term moves (17 × round_up(2·maxBlock, 16) × 4 B).

## 3. Access-pattern characterization

All ring traffic is **dual sequential streams per buffer** — one read tap
walking forward plus one write head walking forward. That is prefetch-
friendly by construction: each stream has stride-1 locality, and the
hardware prefetcher locks onto it within a few accesses. The diffuser's 32
concurrent streams (8 sections × stereo × read+write) are inherent to
8-section stereo Schroeder, not an allocation defect; the L/R rings keep
distinct prime lengths deliberately (a shared ring would make every access
stride-2 and destroy the contiguous 4-wide load/store in `chunk_`).

The historical per-sample costs were **overhead and serial-chain latency,
not DRAM thrash**: per-sample `windowPtr` branching, horizontal-sum latency
on the critical path, single-float write+mirror pairs, and a per-sample sat
switch (fixed in C3 by chunked processing over the loop-carried distance
`Lc ≤ D − 6`), plus stage-split scratch traffic (trimmed in C10 by loop
fusion, −9% mean on the engine row). TLB pressure is real but bounded: at
block 512 the chain touches ~60 4 KB pages per block against a 64-entry L1
dTLB, so some accesses fall to the L2 STLB (~7 cycles) — a small, measured-
flat cost (C9/C9b benches were neutral within noise).

## 4. Known idle capacity

`SimdSVF` runs 2 live lanes of a 4-wide register (stereo packed into lanes
0–1; lanes 2–3 are clamped-zero passengers). That is a candidate **only** if
a future 4-band split needs the other two lanes; do not re-pack for its own
sake. Bench before touching — the SVF cascade is already the second-largest
stage cost and a re-pack risks regressing the common stereo case.

## 5. Standing constraints (unchanged by this audit)

- `process()` performs no allocation, locks, syscalls, or unbounded loops;
  every buffer is sized in `prepare()` (RT-safety house rule).
- No allocator can improve steady-state throughput — there are no RT
  allocations to accelerate. The arena exists for layout and
  instrumentation, which is why the chowdsp-derived pool/chained/STL
  allocators were all rejected (see the plan's allocator table).
