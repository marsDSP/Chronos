#pragma once

#ifndef CHRONOS_NOISE_GENERATOR_ENGINE_H
#define CHRONOS_NOISE_GENERATOR_ENGINE_H

// ============================================================================
//  noise_generator_engine.h
// ----------------------------------------------------------------------------
//  Noise generator for the Ornstein-Uhlenbeck mean reversion algorithm.
// ============================================================================

#include <JuceHeader.h>
#include <array>

namespace MarsDSP::DSP::Modulation
{
    // Forward-declared helpers; defined inside an inline namespace below.
    template <typename T>
    struct NoiseGeneratorHelpersNumericTraits
    {
        using NumericType = T;
    };

    namespace NoiseGeneratorHelpers
    {
        template <typename T>
        T uniformZeroToOne (Random&) noexcept;

        template <>
        inline double uniformZeroToOne (Random& r) noexcept
        {
            return r.nextDouble();
        }

        template <>
        inline float uniformZeroToOne (Random& r) noexcept
        {
            return r.nextFloat();
        }

        template <typename T>
        struct UniformCenteredNoiseGeneratorFunctor
        {
            T operator() (size_t /*ch*/, Random& r) const noexcept
            {
                return static_cast<T>(2) * uniformZeroToOne<T>(r)
                     - static_cast<T>(1);
            }
        };

        template <typename T>
        struct NormalNoiseGeneratorFunctor
        {
            T operator() (size_t /*ch*/, Random& r) const noexcept
            {
                // Box-Muller transform.
                const T radius = std::sqrt(static_cast<T>(-2) * std::log(static_cast<T>(1) - uniformZeroToOne<T>(r)));
                const T theta = MathConstants<T>::twoPi * uniformZeroToOne<T>(r);
                return radius * std::sin(theta) / MathConstants<T>::sqrt2;
            }
        };

        template <typename T, typename ProcessContext, typename FunctorType>
        void fillContextWithRandomFunctor (const ProcessContext& context, Random& r,
                                           FunctorType randFunc) noexcept
        {
            auto&& outBlock     = context.getOutputBlock();
            const auto len      = outBlock.getNumSamples();
            const auto numCh    = outBlock.getNumChannels();

            for (size_t ch = 0; ch < numCh; ++ch)
            {
                auto* dst = outBlock.getChannelPointer(ch);
                for (size_t i = 0; i < len; ++i)
                    dst[i] = randFunc(ch, r);
            }
        }
    }

    // ----------------------------------------------------------------------------
    //  NoiseGeneratorEngine<SampleType>
    //
    //  Block-based noise source with configurable gain (via the underlying
    //  juce::dsp::Gain<> base class). Supports uniform, normal (Gaussian via
    //  Box-Muller) and pink (Voss algorithm) noise types.
    // ----------------------------------------------------------------------------
    template <typename SampleType>
    class NoiseGeneratorEngine : public dsp::Gain<SampleType>
    {
    public:
        using NumericType = SampleType;

        enum NoiseGeneratorType
        {
            Uniform,
            Normal,
            Pink
        };

        NoiseGeneratorEngine() = default;

        void setNoiseType (NoiseGeneratorType newType) noexcept
        {
            noiseType = newType;
        }

        [[nodiscard]] NoiseGeneratorType getNoiseType() const noexcept
        {
            return noiseType;
        }

        void prepare (const dsp::ProcessSpec& spec) noexcept
        {
            dsp::Gain<SampleType>::prepare(spec);

            perChannelRandomSampleBlockStorage.allocate(
                spec.numChannels * spec.maximumBlockSize * sizeof(SampleType),
                true);
            perChannelRandomSampleBlock = dsp::AudioBlock<SampleType>(
                perChannelRandomSampleBlockStorage,
                spec.numChannels, spec.maximumBlockSize);
            perChannelRandomSampleBlock.clear();

            perChannelGainStorageBlockStorage.allocate(
                spec.numChannels * spec.maximumBlockSize * sizeof(SampleType),
                true);
            perChannelGainStorageBlock = dsp::AudioBlock<SampleType>(
                perChannelGainStorageBlockStorage,
                spec.numChannels, spec.maximumBlockSize);
            perChannelGainStorageBlock.clear();

            pinkNoiseVossGenerator.resetForNumberOfChannels(spec.numChannels);
        }

        void reset() noexcept
        {
            dsp::Gain<SampleType>::reset();
        }

        void setDeterministicSeed (int64 newSeed)
        {
            hostRandomEngine.setSeed(newSeed);
        }

        template <typename ProcessContext>
        void process (const ProcessContext& context) noexcept
        {
            if (context.isBypassed) return;

            auto&& outBlock = context.getOutputBlock();
            auto&& inBlock  = context.getInputBlock();
            const auto len  = outBlock.getNumSamples();

            auto randSubBlock = perChannelRandomSampleBlock.getSubBlock(0, len);
            dsp::ProcessContextReplacing<SampleType> randContext(randSubBlock);

            if (noiseType == Uniform)
                NoiseGeneratorHelpers::fillContextWithRandomFunctor<SampleType>(
                    randContext, hostRandomEngine,
                    NoiseGeneratorHelpers::UniformCenteredNoiseGeneratorFunctor<SampleType>());
            else if (noiseType == Normal)
                NoiseGeneratorHelpers::fillContextWithRandomFunctor<SampleType>(
                    randContext, hostRandomEngine,
                    NoiseGeneratorHelpers::NormalNoiseGeneratorFunctor<SampleType>());
            else if (noiseType == Pink)
                NoiseGeneratorHelpers::fillContextWithRandomFunctor<SampleType>(
                    randContext, hostRandomEngine, pinkNoiseVossGenerator);

            // Apply the gain (from juce::dsp::Gain<> base) to the random block.
            dsp::ProcessContextReplacing<SampleType> gainContext(randSubBlock);
            dsp::Gain<SampleType>::process(gainContext);

            if (context.usesSeparateInputAndOutputBlocks())
                outBlock.copyFrom(inBlock);

            outBlock += perChannelRandomSampleBlock;
        }

        void processBlock (dsp::AudioBlock<SampleType>& block) noexcept
        {
            process(juce::dsp::ProcessContextReplacing<SampleType>{ block });
        }

    private:
        // Hide the scalar overload exposed by juce::dsp::Gain.
        SampleType processSample (SampleType) { return SampleType{0}; }

        NoiseGeneratorType noiseType { Normal };
        Random             hostRandomEngine;

        // Voss / Firthering / "The Voss algorithm" pink noise generator.
        //  http://www.firstpr.com.au/dsp/pink-noise/
        template <size_t QualityBits = 8>
        struct PinkNoiseVossGeneratorFunctor
        {
            std::vector<int> frameIndexPerChannel;
            std::vector<std::array<SampleType, QualityBits>> valuesPerChannel;

            void resetForNumberOfChannels (size_t numberOfChannels)
            {
                frameIndexPerChannel.assign(numberOfChannels, -1);
                valuesPerChannel.clear();
                for (size_t ch = 0; ch < numberOfChannels; ++ch)
                {
                    std::array<SampleType, QualityBits> v;
                    v.fill(static_cast<SampleType>(0));
                    valuesPerChannel.push_back(v);
                }
            }

            SampleType operator() (size_t ch, Random& r) noexcept
            {
                const int lastFrame = frameIndexPerChannel[ch];
                frameIndexPerChannel[ch]++;
                if (frameIndexPerChannel[ch] >= (1 << QualityBits))
                    frameIndexPerChannel[ch] = 0;
                const int diff = lastFrame ^ frameIndexPerChannel[ch];

                auto sum = static_cast<SampleType>(0);
                for (size_t i = 0; i < QualityBits; i++)
                {
                    if (diff & (1 << i))
                    {
                        valuesPerChannel[ch][i] =
                            NoiseGeneratorHelpers::uniformZeroToOne<SampleType>(r)
                            - static_cast<SampleType>(0.5);
                    }
                    sum += valuesPerChannel[ch][i];
                }
                return sum * oneOverEight;
            }

            const SampleType oneOverEight = static_cast<SampleType>(1.0 / 8.0);
        };

        PinkNoiseVossGeneratorFunctor<> pinkNoiseVossGenerator;

        HeapBlock<char>               perChannelRandomSampleBlockStorage;
        dsp::AudioBlock<SampleType>   perChannelRandomSampleBlock;

        HeapBlock<char>               perChannelGainStorageBlockStorage;
        dsp::AudioBlock<SampleType>   perChannelGainStorageBlock;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NoiseGeneratorEngine)
    };
}
#endif
