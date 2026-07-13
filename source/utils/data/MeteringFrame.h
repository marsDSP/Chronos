#pragma once

#ifndef CHRONOS_METERINGFRAME_H
#define CHRONOS_METERINGFRAME_H

#include <cstdint>
// =====================================================================
//  MeteringFrame
// ---------------------------------------------------------------------
//  snap of a single processBlock's worth of metering data pushed
//  through SPSC. Entire struct is trivially copyable. No hot path alloc.
// =====================================================================
namespace MarsDSP::Utils
{
    struct MeteringFrame
    {
        float outputPeakLeft  = 0.0f;
        float outputPeakRight = 0.0f;
        float duckerBlockEndGain = 1.0f;
        std::uint64_t blockIndex = 0;
    };
}
#endif