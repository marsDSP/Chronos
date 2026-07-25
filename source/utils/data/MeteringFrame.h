#pragma once

#ifndef CHRONOS_METERINGFRAME_H
#define CHRONOS_METERINGFRAME_H

#include <cstdint>

namespace MarsDSP::Utils::Data
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