#include "ChronosProcessor.h"
#include "ChronosEditor.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace {
    // State schema version written into every saved state tree.
    constexpr int kStateVersion = 2;
    // Cap on the repeat count for the tail length above self-oscillation.
    constexpr int kMaxTailRepeats = 240;
    // Ring-down margin in samples, added to the delay repeat tail.
    constexpr int kMargin = 32768;
    // Constant latency budget of the saturator stage.
    constexpr int kAlignBudget = MarsDSP::Align::SaturatorAlign::kBudget;
}
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
    // Clamp the feedback before the logarithm to stay finite at zero.
    const double g = std::max(static_cast<double>(parameters.getRawFeedback()), 1e-4);
    const double n = (g >= 1.0)
        ? static_cast<double>(kMaxTailRepeats)
        : std::ceil(-3.0 / std::log10(g));
    const double repeatTail = std::min(delaySeconds * n, 60.0);
    return repeatTail + static_cast<double>(kAlignBudget + kMargin) / sr;
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

void ChronosProcessor::reset()
{
    engine.reset();
    parameters.reset();
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
    ValueTree state = apvts.copyState();
    state.setProperty("version", kStateVersion, nullptr);
    copyXmlToBinary(*state.createXml(), destData);
}

void ChronosProcessor::setStateInformation(const void *data, int sizeInBytes)
{
    std::unique_ptr xml(getXmlFromBinary(data, sizeInBytes));
    if (xml == nullptr || !xml->hasTagName(apvts.state.getType()))
        return;
    ValueTree state(ValueTree::fromXml(*xml));
    const int fileVersion = static_cast<int>(state.getProperty("version"));
    if (fileVersion < kStateVersion)
        migrateState_(state, fileVersion);
    state.setProperty("version", kStateVersion, nullptr);
    apvts.replaceState(state);
}

void ChronosProcessor::migrateState_(ValueTree& state, int fromVersion)
{
    if (fromVersion < 2)
    {
        // A version-1 or absent state needs no change yet.
        // Add the clamps to the new legal ranges when the ranges change.
        ignoreUnused(state);
    }
}

//==============================================================================
AudioProcessor * JUCE_CALLTYPE createPluginFilter()
{
    return new ChronosProcessor();
}
