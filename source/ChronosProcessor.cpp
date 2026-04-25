#include "ChronosProcessor.h"
#include "ChronosEditor.h"
#include "PluginParameters.h"
#include "utils/helpers/tempo_sync.h"
//==============================================================================
ChronosProcessor::ChronosProcessor() : AudioProcessor (BusesProperties()
                       .withInput  ("Input",  AudioChannelSet::stereo(), true)
                       .withOutput ("Output", AudioChannelSet::stereo(), true)),
                       apvts(*this, nullptr, "Parameters", createParameterLayout())
{
    xorshiftL = 1.0;
    while (xorshiftL < 16386)
        xorshiftL = rand() * UINT32_MAX;

    xorshiftR = 1.0;
    while (xorshiftR < 16386)
        xorshiftR = rand() * UINT32_MAX;
}

ChronosProcessor::~ChronosProcessor() = default;
//=============================================================================
const String ChronosProcessor::getName() const
{
    return JucePlugin_Name;
}
bool ChronosProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}
bool ChronosProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}
bool ChronosProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}
double ChronosProcessor::getTailLengthSeconds() const
{
    // Forward the DelayEngine's ringout estimate so hosts know when the
    // feedback tail has decayed and the plugin can be released.
    const double sr = getSampleRate();
    if (sr <= 0.0) return 0.0;
    return static_cast<double>(delay.ringoutSamples()) / sr;
}
int ChronosProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}
int ChronosProcessor::getCurrentProgram()
{
    return 0;
}
void ChronosProcessor::setCurrentProgram (int index)
{
    ignoreUnused (index);
}
const String ChronosProcessor::getProgramName (int index)
{
    ignoreUnused (index);
    return {};
}
void ChronosProcessor::changeProgramName (int index, const String& newName)
{
    ignoreUnused (index, newName);
}
//=============================================================================
AudioProcessorValueTreeState::ParameterLayout ChronosProcessor::createParameterLayout()
{
    using namespace Chronos::ParamID;
    AudioProcessorValueTreeState::ParameterLayout layout;

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kDelayTime, 1), "Delay Time",
        NormalisableRange(5.0f, 5000.0f, 0.1f, 0.3f), 200.0f,
        AudioParameterFloatAttributes().withLabel("ms")));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kMix, 1), "Mix",
        NormalisableRange(0.0f, 1.0f, 0.01f), 0.5f,
        AudioParameterFloatAttributes().withLabel("%")));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kFeedback, 1), "Feedback",
        NormalisableRange(0.0f, 0.99f, 0.01f), 0.3f,
        AudioParameterFloatAttributes().withLabel("%")));

    // Feedback-path low-cut (highpass) corner frequency.
    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kLowCut, 1), "Low Cut",
        NormalisableRange(20.0f, 20000.0f, 1.0f, 0.3f), 20.0f,
        AudioParameterFloatAttributes().withLabel("Hz")));

    // Feedback-path high-cut (lowpass) corner frequency.
    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kHighCut, 1), "High Cut",
        NormalisableRange(20.0f, 20000.0f, 1.0f, 0.3f), 20000.0f,
        AudioParameterFloatAttributes().withLabel("Hz")));

    // Stereo crossfeed (ping-pong amount). 0 = none, 1 = full swap.
    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kCrossfeed, 1), "Crossfeed",
        NormalisableRange(0.0f, 1.0f, 0.01f), 0.0f,
        AudioParameterFloatAttributes().withLabel("%")));

    layout.add(std::make_unique<AudioParameterBool>(
        ParameterID(kMono, 1), "Mono", false));

    layout.add(std::make_unique<AudioParameterBool>(
        ParameterID(kBypass, 1), "Bypass", false));

    // ---- ChronosReverb (unified post-delay reverb) --------------------
    // All of these map one-to-one onto DelayEngine::setReverb*Param,
    // which in turn drives the embedded ChronosReverbStereoProcessor.
    // Defaults are picked so the reverb is musically useful out of the
    // box (medium hall, low HF damping, moderate modulation) while the
    // mix defaults to 0 (post-delay send off until the user dials in).
    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kReverbMix, 1), "Reverb Mix",
        NormalisableRange(0.0f, 1.0f, 0.01f), 0.0f,
        AudioParameterFloatAttributes().withLabel("%")));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kReverbRoomSize, 1), "Reverb Room Size",
        NormalisableRange(-2.0f, 2.0f, 0.01f), 0.0f));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kReverbDecayTime, 1), "Reverb Decay Time",
        NormalisableRange(-4.0f, 6.0f, 0.01f), 0.75f));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kReverbPredelay, 1), "Reverb Predelay",
        NormalisableRange(-8.0f, 1.0f, 0.01f), -4.0f));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kReverbDiffusion, 1), "Reverb Diffusion",
        NormalisableRange(0.0f, 1.0f, 0.01f), 1.0f,
        AudioParameterFloatAttributes().withLabel("%")));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kReverbBuildup, 1), "Reverb Buildup",
        NormalisableRange(0.0f, 1.0f, 0.01f), 1.0f,
        AudioParameterFloatAttributes().withLabel("%")));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kReverbModulation, 1), "Reverb Modulation",
        NormalisableRange(0.0f, 1.0f, 0.01f), 0.5f,
        AudioParameterFloatAttributes().withLabel("%")));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kReverbHighFrequencyDamping, 1), "Reverb HF Damping",
        NormalisableRange(0.0f, 1.0f, 0.01f), 0.2f,
        AudioParameterFloatAttributes().withLabel("%")));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kReverbLowFrequencyDamping, 1), "Reverb LF Damping",
        NormalisableRange(0.0f, 1.0f, 0.01f), 0.2f,
        AudioParameterFloatAttributes().withLabel("%")));

    // Single toggle that bypasses the reverb send cleanly via the
    // engine's CrossfadeBypassEngine. Off by default.
    layout.add(std::make_unique<AudioParameterBool>(
        ParameterID(kReverbBypass, 1), "Reverb Bypass", false));

    // ---- Wow and flutter ----------------------------------------------
    // Wow: slow cosine drift with per-block random rate perturbation.
    // Flutter: sum of three cosines (f, 2f, 3f). Flutter has its own
    // on/off switch because it is meaningful to crank wow without
    // adding the capstan wobble on top.
    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kWowRate, 1), "Wow Rate",
        NormalisableRange(0.0f, 1.0f, 0.01f), 0.5f));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kWowDepth, 1), "Wow Depth",
        NormalisableRange(0.0f, 1.0f, 0.01f), 0.0f,
        AudioParameterFloatAttributes().withLabel("%")));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kWowDrift, 1), "Wow Drift",
        NormalisableRange(0.0f, 1.0f, 0.01f), 0.0f,
        AudioParameterFloatAttributes().withLabel("%")));

    layout.add(std::make_unique<AudioParameterBool>(
        ParameterID(kFlutterOnOff, 1), "Flutter", false));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kFlutterRate, 1), "Flutter Rate",
        NormalisableRange(0.0f, 1.0f, 0.01f), 0.5f));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kFlutterDepth, 1), "Flutter Depth",
        NormalisableRange(0.0f, 1.0f, 0.01f), 0.0f,
        AudioParameterFloatAttributes().withLabel("%")));

    // ---- Bridge ducker (sidechain duck on the wet path) ---------------
    // Five knobs only: bypass, threshold, amount, attack, release.
    // Default: bypassed so existing presets are unaffected.
    layout.add(std::make_unique<AudioParameterBool>(
        ParameterID(kDuckerBypass, 1), "Ducker Bypass", true));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kDuckerThreshold, 1), "Ducker Threshold",
        NormalisableRange(-60.0f, 0.0f, 0.1f), -24.0f,
        AudioParameterFloatAttributes().withLabel("dB")));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kDuckerAmount, 1), "Ducker Amount",
        NormalisableRange(0.0f, 1.0f, 0.01f), 0.5f,
        AudioParameterFloatAttributes().withLabel("%")));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kDuckerAttack, 1), "Ducker Attack",
        NormalisableRange(0.5f, 50.0f, 0.1f, 0.4f), 5.0f,
        AudioParameterFloatAttributes().withLabel("ms")));

    layout.add(std::make_unique<AudioParameterFloat>(
        ParameterID(kDuckerRelease, 1), "Ducker Release",
        NormalisableRange(10.0f, 500.0f, 0.5f, 0.4f), 80.0f,
        AudioParameterFloatAttributes().withLabel("ms")));

    // ---- Tempo sync ---------------------------------------------------
    layout.add(std::make_unique<AudioParameterBool>(
        ParameterID(kSyncEnabled, 1), "Sync", false));

    {
        // Build a StringArray from the constexpr label table in
        // utils/helpers/tempo_sync.h. Default to a straight quarter note.
        StringArray syncIntervalChoiceLabels;
        for (int intervalIndex = 0;
             intervalIndex < MarsDSP::Utils::getNumberOfTempoSyncIntervals();
             ++intervalIndex)
        {
            const auto labelStringView = MarsDSP::Utils::kSyncIntervalDisplayLabels[static_cast<std::size_t>(intervalIndex)];
            syncIntervalChoiceLabels.add(String(labelStringView.data(), labelStringView.size()));
        }
        const int defaultSyncIntervalIndex = static_cast<int>(MarsDSP::Utils::TempoSyncInterval::QuarterNoteStraight);

        layout.add(std::make_unique<AudioParameterChoice>(
            ParameterID(kSyncInterval, 1), "Sync Interval",
            syncIntervalChoiceLabels, defaultSyncIntervalIndex));
    }

    return layout;
}
//==============================================================================
void ChronosProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    dsp::ProcessSpec spec {};
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<uint32>(samplesPerBlock);
    spec.numChannels = static_cast<uint32>(getTotalNumOutputChannels());
    delay.prepare(spec);
}
//=============================================================================
void ChronosProcessor::releaseResources()
{
}
bool ChronosProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}
//=============================================================================
void ChronosProcessor::processBlock (AudioBuffer<float> &buffer, MidiBuffer &midiMessages)
{
    ignoreUnused (buffer, midiMessages);
    ScopedNoDenormals noDenormals;
    ZoneScoped;

    const int numSamples = buffer.getNumSamples();
    if (numSamples == 0)
        return;

    // read parameters (all IDs funnel through Chronos::ParamID)
    using namespace Chronos::ParamID;

    // --- Tempo sync: when the sync toggle is ON, override the delay time
    //     parameter with the musical division mapped to the host BPM. When
    //     the toggle is OFF, the raw "delayTime" parameter is used as-is
    //     (classic DAW plugin behaviour: sync does nothing until enabled).
    const bool syncIsEnabled = apvts.getRawParameterValue(kSyncEnabled)->load() >= 0.5f;
    float resolvedDelayTimeMilliseconds = apvts.getRawParameterValue(kDelayTime)->load();

    if (syncIsEnabled)
    {
        const int syncIntervalChoiceIndex = static_cast<int>(apvts.getRawParameterValue(kSyncInterval)->load());
        const auto syncIntervalEnumValue = static_cast<MarsDSP::Utils::TempoSyncInterval>(syncIntervalChoiceIndex);

        double hostBeatsPerMinute = 120.0;
        if (auto* hostPlayHead = getPlayHead())
        {
            if (const auto hostPositionInfo = hostPlayHead->getPosition())
            {
                if (const auto reportedBpm = hostPositionInfo->getBpm())
                    hostBeatsPerMinute = *reportedBpm;
            }
        }

        const double resolvedMillisecondsFromSync = MarsDSP::Utils::convertTempoSyncIntervalToMilliseconds(
                                                        syncIntervalEnumValue, hostBeatsPerMinute);

        // Clamp into the engine's accepted delay-time range.
        resolvedDelayTimeMilliseconds = std::clamp(static_cast<float>(resolvedMillisecondsFromSync), 5.0f, 5000.0f);
    }

    delay.setDelayTimeParam       (resolvedDelayTimeMilliseconds);
    delay.setMixParam             (apvts.getRawParameterValue(kMix)->load());
    delay.setFeedbackParam        (apvts.getRawParameterValue(kFeedback)->load());
    delay.setLowCutParam          (apvts.getRawParameterValue(kLowCut)->load());
    delay.setHighCutParam         (apvts.getRawParameterValue(kHighCut)->load());
    delay.setCrossfeedParam       (apvts.getRawParameterValue(kCrossfeed)->load());

    // ChronosReverb parameters - unified post-delay reverb.
    delay.setReverbMixParam                 (apvts.getRawParameterValue(kReverbMix)->load());
    delay.setReverbRoomSizeParam            (apvts.getRawParameterValue(kReverbRoomSize)->load());
    delay.setReverbDecayTimeParam           (apvts.getRawParameterValue(kReverbDecayTime)->load());
    delay.setReverbPredelayParam            (apvts.getRawParameterValue(kReverbPredelay)->load());
    delay.setReverbDiffusionParam           (apvts.getRawParameterValue(kReverbDiffusion)->load());
    delay.setReverbBuildupParam             (apvts.getRawParameterValue(kReverbBuildup)->load());
    delay.setReverbModulationParam          (apvts.getRawParameterValue(kReverbModulation)->load());
    delay.setReverbHighFrequencyDampingParam(apvts.getRawParameterValue(kReverbHighFrequencyDamping)->load());
    delay.setReverbLowFrequencyDampingParam (apvts.getRawParameterValue(kReverbLowFrequencyDamping)->load());
    delay.setReverbBypassedParam            (apvts.getRawParameterValue(kReverbBypass)->load() >= 0.5f);

    // Wow and flutter: push knob values into the engine every block.
    delay.setWowRateParam     (apvts.getRawParameterValue(kWowRate)->load());
    delay.setWowDepthParam    (apvts.getRawParameterValue(kWowDepth)->load());
    delay.setWowDriftParam    (apvts.getRawParameterValue(kWowDrift)->load());
    delay.setFlutterOnOffParam(apvts.getRawParameterValue(kFlutterOnOff)->load() >= 0.5f);
    delay.setFlutterRateParam (apvts.getRawParameterValue(kFlutterRate)->load());
    delay.setFlutterDepthParam(apvts.getRawParameterValue(kFlutterDepth)->load());

    // Bridge ducker: 5-knob sidechain duck on the wet taps.
    delay.setDuckerBypassedParam (apvts.getRawParameterValue(kDuckerBypass)   ->load() >= 0.5f);
    delay.setDuckerThresholdParam(apvts.getRawParameterValue(kDuckerThreshold)->load());
    delay.setDuckerAmountParam   (apvts.getRawParameterValue(kDuckerAmount)   ->load());
    delay.setDuckerAttackParam   (apvts.getRawParameterValue(kDuckerAttack)   ->load());
    delay.setDuckerReleaseParam  (apvts.getRawParameterValue(kDuckerRelease)  ->load());

    delay.setMono                 (apvts.getRawParameterValue(kMono)  ->load() >= 0.5f);
    delay.setBypassed             (apvts.getRawParameterValue(kBypass)->load() >= 0.5f);

    // Wrap the host's JUCE AudioBuffer in an AlignedSIMDBufferView so the
    // DelayEngine can consume it through the xsimd-aligned buffer API
    // without going through juce::dsp::AudioBlock.
    const MarsDSP::DSP::AlignedBuffers::AlignedSIMDBufferView inOutBlockView(buffer.getArrayOfWritePointers(),
                                                                             buffer.getNumChannels(),
                                                                             buffer.getNumSamples());
    delay.process(inOutBlockView, numSamples);

    // advance dither state
    xorshiftL ^= xorshiftL << 13;
    xorshiftL ^= xorshiftL >> 17;
    xorshiftL ^= xorshiftL << 5;

    xorshiftR ^= xorshiftR << 13;
    xorshiftR ^= xorshiftR >> 17;
    xorshiftR ^= xorshiftR << 5;

#if JUCE_DEBUG
    MarsDSP::overloaded(buffer);
#endif
    FrameMark;
}
//==============================================================================
bool ChronosProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}
AudioProcessorEditor* ChronosProcessor::createEditor()
{
    return new ChronosEditor(*this);
}
//==============================================================================
void ChronosProcessor::getStateInformation(MemoryBlock &destData)
{
    // Stamp the current state-tree version so a future build can migrate
    // this state forward if the schema changes.
    auto state = apvts.copyState();
    state.setProperty(Chronos::kStateVersionProperty, Chronos::kPluginStateVersion, nullptr);

    std::unique_ptr xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void ChronosProcessor::setStateInformation(const void *data, int sizeInBytes)
{
    std::unique_ptr xml(getXmlFromBinary(data, sizeInBytes));
    if (xml == nullptr || !xml->hasTagName(apvts.state.getType()))
        return;

    auto tree = ValueTree::fromXml(*xml);

    // Pre-versioning builds stored no version property; treat as v0.
    const int fromVersion = tree.getProperty(Chronos::kStateVersionProperty, 0);
    Chronos::migrateStateTree(tree, fromVersion);

    apvts.replaceState(tree);
}
//==============================================================================
AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new ChronosProcessor();
}
