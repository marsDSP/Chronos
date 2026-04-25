#pragma once

// =====================================================================
//  MeteringFrame
// ---------------------------------------------------------------------
//  Snapshot of a single processBlock's worth of metering data, pushed
//  from the DSP thread to the UI thread through the project's SPSC
//  queue (see utils/memory/spsc_queue.h).
//
//  The struct is deliberately a trivially-copyable POD: the queue moves
//  whole frames through placement new / std::move, and we never want a
//  metering write on the audio thread to allocate, lock, or otherwise
//  block.
// =====================================================================

#include <cstdint>

namespace Chronos
{
    struct MeteringFrame
    {
        // Linear peak magnitudes of the post-FX wet output for the
        // block, per channel.  Range [0, +inf) but practically [0, 1.5].
        float outputPeakLeft  = 0.0f;
        float outputPeakRight = 0.0f;

        // Bridge ducker block-end gain.  1.0 = no attenuation,
        // 0.0 = signal fully shunted.  When the ducker is bypassed
        // this is reported as 1.0 so the UI always shows "no GR".
        float duckerBlockEndGain = 1.0f;

        // Monotonic block index, useful for the UI to detect dropped /
        // skipped frames if the queue overflows momentarily.
        std::uint64_t blockIndex = 0;
    };
}
