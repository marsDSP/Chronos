#pragma once

#include "Converters.h"

namespace MarsDSP {

    inline const juce::ParameterID oversamplingChoiceID { "oversample", 1 };
    static constexpr const char* oversamplingChoiceName = "Oversampling Choice";

    inline const juce::ParameterID timeParamID { "time", 1 };
    static constexpr const char* timeParamIDName = "Time";

    inline const juce::ParameterID regenParamID { "regen", 1 };
    static constexpr const char* regenParamIDName = "Regen";

    inline const juce::ParameterID freqParamID { "freq", 1 };
    static constexpr const char* freqParamIDName = "Freq";

    inline const juce::ParameterID resoParamID { "reso", 1 };
    static constexpr const char* resoParamIDName = "Reso";

    inline const juce::ParameterID flutterParamID { "flutter", 1 };
    static constexpr const char* flutterParamIDName = "Flutter";

    inline const juce::ParameterID drywetParamID { "drywet", 1 };
    static constexpr const char* drywetParamIDName = "DryWet";

    inline const juce::ParameterID bypassParamID { "bypass", 1 };
    static constexpr const char* bypassParamIDName = "Bypass";

    inline const juce::StringArray items
    {
        "OFF", "2x", "4x", "8x", "16x"
    };

    class Parameters
    {
    public:

        explicit Parameters(juce::AudioProcessorValueTreeState& vts)
        {
            castParameter (vts, oversamplingChoiceID, oversample);
            castParameter (vts, timeParamID, time);
            castParameter (vts, regenParamID, regen);
            castParameter (vts, freqParamID, freq);
            castParameter (vts, resoParamID, reso);
            castParameter (vts, flutterParamID, flutter);
            castParameter (vts, drywetParamID, drywet);
            castParameter (vts, bypassParamID, bypass);
        }

        static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
        {
            juce::AudioProcessorValueTreeState::ParameterLayout layout;

            layout.add(std::make_unique<juce::AudioParameterChoice>
                (oversamplingChoiceID, oversamplingChoiceName, items, 0));

            layout.add(std::make_unique<juce::AudioParameterFloat>
                (timeParamID, timeParamIDName, juce::NormalisableRange<float>
                    {minDelayTime, maxDelayTime, 0.001f, 0.25f}, 0.0f,
                juce::AudioParameterFloatAttributes()
                .withStringFromValueFunction(Converter::stringFromMilliseconds)
                .withValueFromStringFunction(Converter::millisecondsFromString)));

            layout.add(std::make_unique<juce::AudioParameterFloat>
                (regenParamID, regenParamIDName, juce::NormalisableRange<float>
                    {0.0f, 100.0f, 1.0f}, 0.0f,
                juce::AudioParameterFloatAttributes()
                .withStringFromValueFunction(Converter::stringFromPercent)));

            layout.add(std::make_unique<juce::AudioParameterFloat>
                (freqParamID, freqParamIDName, juce::NormalisableRange<float>
                    {0.0f, 100.0f, 1.0f}, 0.0f,
                juce::AudioParameterFloatAttributes()
                .withStringFromValueFunction(Converter::stringFromHz)
                .withValueFromStringFunction(Converter::hzFromString)));

            layout.add(std::make_unique<juce::AudioParameterFloat>
                (resoParamID, resoParamIDName, juce::NormalisableRange<float>
                    {0.0f, 100.0f, 1.0f}, 50.0f,
                juce::AudioParameterFloatAttributes()
                .withStringFromValueFunction(Converter::stringFromPercent)));

            layout.add(std::make_unique<juce::AudioParameterFloat>
                (flutterParamID, flutterParamIDName, juce::NormalisableRange<float>
                    {0.0f, 100.0f, 1.0f}, 50.0f,
                juce::AudioParameterFloatAttributes()
                .withStringFromValueFunction(Converter::stringFromPercent)));

            layout.add(std::make_unique<juce::AudioParameterFloat>
                (drywetParamID, drywetParamIDName, juce::NormalisableRange<float>
                    {0.0f, 100.0f, 1.0f}, 50.0f,
                juce::AudioParameterFloatAttributes()
                .withStringFromValueFunction(Converter::stringFromPercent)));

            layout.add(std::make_unique<juce::AudioParameterBool>(bypassParamID, bypassParamIDName, false));

            return layout;
        }

        ~Parameters() = default;

        juce::AudioParameterBool* bypass {nullptr};
        juce::AudioParameterChoice* oversample { nullptr };
        juce::AudioParameterFloat* time { nullptr };
        juce::AudioParameterFloat* regen { nullptr };
        juce::AudioParameterFloat* freq { nullptr };
        juce::AudioParameterFloat* reso { nullptr };
        juce::AudioParameterFloat* flutter { nullptr };
        juce::AudioParameterFloat* drywet { nullptr} ;

    private:

        template<typename T>
        static void castParameter(juce::AudioProcessorValueTreeState& vts, const juce::ParameterID& id, T& destination)
        {
            destination = dynamic_cast<T>(vts.getParameter(id.getParamID()));
            jassert(destination);
        }

        static constexpr float minDelayTime = 0.0f;
        static constexpr float maxDelayTime = 500.0f;

    };
} // namespace MarsDSP
