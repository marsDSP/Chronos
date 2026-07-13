#pragma once

#ifndef CHRONOS_OVERLOAD_H
#define CHRONOS_OVERLOAD_H

#include <JuceHeader.h>

namespace MarsDSP::Utils::Helpers
{
    inline void overloaded(AudioBuffer<float> &buf)
    {
        bool warningOne = true;

        for (int ch {}; ch < buf.getNumChannels(); ++ch)
        {
            const float* channelData = buf.getWritePointer(ch);

            for (int smp {}; smp < buf.getNumSamples(); ++smp)
            {
                const float db = channelData[smp];
                bool silence = false;

                if (std::isnan(db))
                {
                    DBG("NaN detected! Silencing!");
                    silence = true;
                }
                else if (std::isinf(db))
                {
                    DBG("INF detected! Silencing!");
                    silence = true;
                }
                else if (db < - 1.0f || db > 1.0f)
                {
                    if (warningOne)
                    {
                        DBG("Sample out of range: " << db << " !!");
                        warningOne = false;
                    }
                }
                if (silence)
                {
                    buf.clear();
                    return;
                }
            }
        }
    }
}
#endif