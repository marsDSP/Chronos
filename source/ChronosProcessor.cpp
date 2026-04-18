#include "ChronosProcessor.h"
#include "ChronosEditor.h"
#include "PluginParameters.h"
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

ChronosProcessor::~ChronosProcessor()
{
}
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
    delay.setDelayTimeParam(apvts.getRawParameterValue(kDelayTime)->load());
    delay.setMixParam     (apvts.getRawParameterValue(kMix)->load());
    delay.setFeedbackParam(apvts.getRawParameterValue(kFeedback)->load());
    delay.setLowCutParam  (apvts.getRawParameterValue(kLowCut)->load());
    delay.setHighCutParam (apvts.getRawParameterValue(kHighCut)->load());
    delay.setCrossfeedParam(apvts.getRawParameterValue(kCrossfeed)->load());
    delay.setMono    (apvts.getRawParameterValue(kMono)  ->load() >= 0.5f);
    delay.setBypassed(apvts.getRawParameterValue(kBypass)->load() >= 0.5f);

    const dsp::AudioBlock<float> block(buffer);
    delay.process(block, numSamples);

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
    return new GenericAudioProcessorEditor(*this);
}
//==============================================================================
void ChronosProcessor::getStateInformation(MemoryBlock &destData)
{
    // Stamp the current state-tree version so a future build can migrate
    // this state forward if the schema changes.
    auto state = apvts.copyState();
    state.setProperty(Chronos::kStateVersionProperty,
                      Chronos::kPluginStateVersion, nullptr);

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
    const int fromVersion = static_cast<int>(
        tree.getProperty(Chronos::kStateVersionProperty, 0));
    Chronos::migrateStateTree(tree, fromVersion);

    apvts.replaceState(tree);
}
//==============================================================================
AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new ChronosProcessor();
}
