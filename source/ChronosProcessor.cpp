#include "ChronosProcessor.h"
#include "ChronosEditor.h"

#include <random>
//==============================================================================
ChronosProcessor::ChronosProcessor() : AudioProcessor(BusesProperties()
    .withInput("Input", AudioChannelSet::stereo(), true)
    .withOutput("Output", AudioChannelSet::stereo(), true)
)
{
    std::random_device rd;
    std::uniform_int_distribution seedDist{16386u, UINT32_MAX};
    engine.setDitherSeeds(seedDist(rd), seedDist(rd));
}

ChronosProcessor::~ChronosProcessor() = default;

//==============================================================================
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
    const double sr = getSampleRate();
    if (sr <= 0.0) return 0.0;
    const double delaySeconds = static_cast<double>(parameters.getDelayMs()) * 0.001;
    constexpr int kMargin = 32768;
    constexpr int kAlignBudget = MarsDSP::Align::SaturatorAlign::kBudget;
    return delaySeconds + static_cast<double>(kMargin) / sr
           + static_cast<double>(kAlignBudget) / sr;
}

int ChronosProcessor::getNumPrograms()
{
    return 1;
}

int ChronosProcessor::getCurrentProgram()
{
    return 0;
}

void ChronosProcessor::setCurrentProgram(int index)
{
    ignoreUnused(index);
}

const String ChronosProcessor::getProgramName(int index)
{
    ignoreUnused(index);
    return {};
}

void ChronosProcessor::changeProgramName(int index, const String &newName)
{
    ignoreUnused(index, newName);
}

//==============================================================================
void ChronosProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    parameters.prepare(sampleRate);
    parameters.reset();

    const int numChannels = getTotalNumInputChannels() > 1 ? 2 : 1;
    engine.prepare(sampleRate, samplesPerBlock, numChannels);
    engine.reset();

    MarsDSP::ChronosEngine::Params p {};
    p.delaySamples = parameters.getDelaySamples();
    p.driveLin = parameters.getRawDriveLin();
    p.mix = parameters.getRawMix();
    p.gainLin = parameters.getRawGainLin();
    p.hpfHz = parameters.getRawHpfHz();
    p.lpfHz = parameters.getRawLpfHz();
    p.bits = parameters.getRawBits();
    p.adaaOrder = parameters.getADAAOrder();
    p.interp = parameters.getInterpolation();
    p.feedback = parameters.getRawFeedback();
    p.dampHz = parameters.getRawDampHz();
    p.crossFeed = parameters.getRawCrossFeed();
    p.loopDrive = parameters.getRawLoopDrive();
    p.loopSatOrder = parameters.getRawLoopSatOrder();
    p.diffusion = parameters.getRawDiffusion();
    p.diffuserSize = parameters.getRawDiffuserSize();
    p.diffModDepth = parameters.getRawDiffModDepth();
    p.diffModRateHz = parameters.getRawDiffModRateHz();
    p.enableDiffuser = parameters.getRawEnableDiffuser();
    engine.resetParams(p);

    // Constant compile-time latency, reported once here
    setLatencySamples(MarsDSP::Align::SaturatorAlign::kBudget);
}

void ChronosProcessor::releaseResources()
{
}

bool ChronosProcessor::isBusesLayoutSupported(const BusesLayout &layouts) const
{
#if JucePlugin_IsMidiEffect
    ignoreUnused(layouts);
    return true;
#else
    if (layouts.getMainOutputChannelSet() != AudioChannelSet::mono()
        && layouts.getMainOutputChannelSet() != AudioChannelSet::stereo())
        return false;

#if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
#endif

    return true;
#endif
}

void ChronosProcessor::processBlock(AudioBuffer<float> &buffer, [[maybe_unused]] MidiBuffer &midiMessages)
{
    ignoreUnused(midiMessages);

    ScopedNoDenormals noDenormals;

    auto totalNumInputChannels = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    // The engine handles bypass internally with a latency-aligned fade.
    parameters.update();
    engine.setBypass(parameters.getBypass());

    const int numSamples = buffer.getNumSamples();
    if (numSamples <= 0) return;

    MarsDSP::ChronosEngine::Params p {};
    p.delaySamples = parameters.getDelaySamples();
    p.driveLin = parameters.getRawDriveLin();
    p.mix = parameters.getRawMix();
    p.gainLin = parameters.getRawGainLin();
    p.hpfHz = parameters.getRawHpfHz();
    p.lpfHz = parameters.getRawLpfHz();
    p.bits = parameters.getRawBits();
    p.adaaOrder = parameters.getADAAOrder();
    p.interp = parameters.getInterpolation();
    p.feedback = parameters.getRawFeedback();
    p.dampHz = parameters.getRawDampHz();
    p.crossFeed = parameters.getRawCrossFeed();
    p.loopDrive = parameters.getRawLoopDrive();
    p.loopSatOrder = parameters.getRawLoopSatOrder();
    p.diffusion = parameters.getRawDiffusion();
    p.diffuserSize = parameters.getRawDiffuserSize();
    p.diffModDepth = parameters.getRawDiffModDepth();
    p.diffModRateHz = parameters.getRawDiffModRateHz();
    p.enableDiffuser = parameters.getRawEnableDiffuser();
    engine.setParams(p);

    float *io[2] = {
        buffer.getWritePointer(0),
        totalNumInputChannels > 1 ? buffer.getWritePointer(1) : nullptr
    };
    engine.process(io, totalNumInputChannels, numSamples);
}

//==============================================================================
bool ChronosProcessor::hasEditor() const
{
    return true;
}

AudioProcessorEditor *ChronosProcessor::createEditor()
{
    return new GenericAudioProcessorEditor(*this);
}

//==============================================================================
void ChronosProcessor::getStateInformation(MemoryBlock &destData)
{
    copyXmlToBinary(*apvts.copyState().createXml(), destData);
}

void ChronosProcessor::setStateInformation(const void *data, int sizeInBytes)
{
    std::unique_ptr xml(getXmlFromBinary(data, sizeInBytes));
    if (xml != nullptr && xml->hasTagName(apvts.state.getType()))
    {
        apvts.replaceState(ValueTree::fromXml(*xml));
    }
}

//==============================================================================
AudioProcessor * JUCE_CALLTYPE createPluginFilter()
{
    return new ChronosProcessor();
}
