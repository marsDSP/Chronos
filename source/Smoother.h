#pragma once

#include "Parameters.h"

namespace MarsDSP {

    class Smoother {

    public:

        explicit Smoother(const Parameters& p) : params(p) {}

        void prepare(const juce::dsp::ProcessSpec& spec) noexcept
        {
            constexpr double duration = 0.02;
            const int steps = static_cast<int>(spec.sampleRate * duration);
            auto resetAll = [steps](auto& smootherArray)
            {
                for (auto& smoother : smootherArray)
                    smoother.reset(steps);
            };

            resetAll(timeSmoother);
            resetAll(regenSmoother);
            resetAll(freqSmoother);
            resetAll(resoSmoother);
            resetAll(flutterSmoother);
            resetAll(drywetSmoother);
        }

        void reset() noexcept
        {
            time = 0.0f;
            const auto timeVal = params.time->get() / 500.0f;
            for (auto& smoother : timeSmoother)
                smoother.setCurrentAndTargetValue(timeVal);

            regen = 0.0f;
            for (auto& smoother : regenSmoother)
                smoother.setCurrentAndTargetValue(params.regen->get() * 0.01f);

            freq = 0.0f;
            for (auto& smoother : freqSmoother)
                smoother.setCurrentAndTargetValue(params.freq->get() * 0.01f);

            reso = 1.0f;
            for (auto& smoother : resoSmoother)
                smoother.setCurrentAndTargetValue(params.reso->get() * 0.01f);

            flutter = 0.0f;
            for (auto& smoother : flutterSmoother)
                smoother.setCurrentAndTargetValue(params.flutter->get() * 0.01f);

            drywet = 0.0f;
            for (auto& smoother : drywetSmoother)
                smoother.setCurrentAndTargetValue(params.drywet->get() * 0.01f);

        }

        void update() noexcept
        {
            const float TimeMs = params.time->get();
            const float newTime = TimeMs / 500.0f;
            for (auto& smoother : timeSmoother)
                smoother.setTargetValue(newTime);

            const float newRegen = params.regen->get() * 0.01f; // 0..1
            for (auto& smoother : regenSmoother)
                smoother.setTargetValue(newRegen);

            const float newFreq = params.freq->get() * 0.01f;
            for (auto& smoother : freqSmoother)
                smoother.setTargetValue(newFreq);

            const float newReso = params.reso->get() * 0.01f; // -1..1
            for (auto& smoother : resoSmoother)
                smoother.setTargetValue(newReso);

            const float newFlutter = params.flutter->get() * 0.01f; // -1..1
            for (auto& smoother : flutterSmoother)
                smoother.setTargetValue(newFlutter);

            const float newDryWet = params.drywet->get() * 0.01f; // -1..1
            for (auto& smoother : drywetSmoother)
                smoother.setTargetValue(newDryWet);

            bypassed = params.bypass->get();
            oversample = params.oversample->getIndex();
        }

        void smoothen() noexcept
        {
            auto smoothen = [](auto& smootherArray)
            {
                for (auto& smoother : smootherArray)
                    smoother.getNextValue();
            };

            smoothen(timeSmoother);
            smoothen(regenSmoother);
            smoothen(freqSmoother);
            smoothen(resoSmoother);
            smoothen(flutterSmoother);
            smoothen(drywetSmoother);
        }

        std::vector<std::array<juce::LinearSmoothedValue<float>, 2>*> getSmoother() noexcept
        {
            return { &timeSmoother,
                     &regenSmoother,
                     &freqSmoother,
                     &resoSmoother,
                     &flutterSmoother,
                     &drywetSmoother };
        }

        enum class SmootherUpdateMode
        {
            initialize,
            liveInRealTime
        };

        void setSmoother(int numSamplesToSkip, SmootherUpdateMode init) noexcept
        {
            juce::ignoreUnused(init);

            auto skipArray = [numSamplesToSkip](auto& smootherArray)
            {
                for (auto& s : smootherArray)
                    s.skip(numSamplesToSkip);
            };

            skipArray(timeSmoother);
            skipArray(regenSmoother);
            skipArray(freqSmoother);
            skipArray(resoSmoother);
            skipArray(flutterSmoother);
            skipArray(drywetSmoother);
        }

        float getTime    (size_t channel = 0) noexcept { return timeSmoother[channel].getNextValue(); }
        float getRegen   (size_t channel = 0) noexcept { return regenSmoother[channel].getNextValue(); }
        float getFreq    (size_t channel = 0) noexcept { return freqSmoother[channel].getNextValue(); }
        float getReso    (size_t channel = 0) noexcept { return resoSmoother[channel].getNextValue(); }
        float getFlutter (size_t channel = 0) noexcept { return flutterSmoother[channel].getNextValue(); }
        float getDryWet  (size_t channel = 0) noexcept { return drywetSmoother[channel].getNextValue(); }

    private:

        // we don't need to copy, just reference
        const Parameters& params;

        float time    { 0.0f };
        float regen    { 0.0f };
        float freq    { 0.0f };
        float reso { 0.0f };
        float flutter   { 0.0f };
        float drywet { 0.0f };
        int oversample { 0 };
        bool bypassed  { false };

        std::array<juce::LinearSmoothedValue<float>, 2>

        timeSmoother,
        regenSmoother,
        freqSmoother,
        resoSmoother,
        flutterSmoother,
        drywetSmoother;

    };
} // namespace MarsDSP
