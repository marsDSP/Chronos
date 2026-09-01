#include "ChronosProcessor.h"
#include "ChronosEditor.h"
#include "utils/helpers/TempoSync.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>

namespace
{
    // State schema version written into every saved state tree.
    constexpr int kStateVersion = 5;
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
    // Use the synced delay when tempo sync is on. The knob value alone
    // truncates the tail.
    const auto [delL, delR] = computeDelaySamples_();
    const double delaySeconds = static_cast<double>(std::max(delL, delR)) / sr;
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
    return programName_;
}

void ChronosProcessor::changeProgramName(int index, const String &newName)
{
    ignoreUnused(index);
    programName_ = newName;
}

//==============================================================================
void ChronosProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    parameters.prepare(sampleRate);
    parameters.reset();

    const int numChannels = getTotalNumInputChannels() > 1 ? 2 : 1;
    engine.prepare(sampleRate, samplesPerBlock, numChannels);
    engine.reset();

    MarsDSP::ChronosEngine::Params p{};
    const auto [delL, delR] = computeDelaySamples_();
    p.delaySamplesL = delL;
    p.delaySamplesR = delR;
    p.driveLin = parameters.getRawDriveLin();
    p.mix = parameters.getRawMix();
    p.gainLin = parameters.getRawGainLin();
    p.hpfHz = parameters.getRawHpfHz();
    p.lpfHz = parameters.getRawLpfHz();
    p.filterMode = parameters.getRawFilterMode();
    p.bits = parameters.getRawBits();
    p.adaaOrder = parameters.getADAAOrder();
    p.feedback = parameters.getRawFeedback();
    p.dampHz = parameters.getRawDampHz();
    p.loopCutHz = parameters.getRawLoopCutHz();
    p.crossFeed = parameters.getRawCrossFeed();
    p.loopDrive = parameters.getRawLoopDrive();
    p.loopSatOrder = parameters.getRawLoopSatOrder();
    p.diffusion = parameters.getRawDiffusion();
    p.diffuserSize = parameters.getRawDiffuserSize();
    p.diffModDepth = parameters.getRawDiffModDepth();
    p.diffModRateHz = parameters.getRawDiffModRateHz();
    p.enableDiffuser = parameters.getRawEnableDiffuser();
    p.delaySync = parameters.getRawDelaySync();
    p.delayDivision = parameters.getRawDelayDivision();
    p.delayMode = parameters.getRawDelayMode();
    p.delayModDepth = parameters.getRawDelayModDepth();
    p.delayModRateHz = parameters.getRawDelayModRateHz();
    engine.resetParams(p);

    setLatencySamples(MarsDSP::Align::SaturatorAlign::kBudget);
}

std::pair<float, float> ChronosProcessor::computeDelaySamples_() const
{
    if (!parameters.getRawDelaySync())
        return {parameters.getDelaySamplesL(), parameters.getDelaySamplesR()};
    const double ms = MarsDSP::Utils::Helpers::TempoSync::convertChoiceIndexToMilliseconds(
        parameters.getRawDelayDivision(), cachedBpm_.load(std::memory_order_relaxed));
    const double clamped = std::clamp(ms, 1.0, 5000.0);
    const auto s = static_cast<float>(clamped * 0.001 * getSampleRate());
    return {s, s};
}

void ChronosProcessor::releaseResources()
{
    /* let the OS handle this for now */
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

    const auto totalNumInputChannels = getTotalNumInputChannels();
    const auto totalNumOutputChannels = getTotalNumOutputChannels();

    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    parameters.update();
    engine.setBypass(parameters.getBypass());

    // Read the host tempo. Hold the last known BPM when the host gives none.
    if (const auto pos = getPlayHead()->getPosition())
        if (const auto bpm = pos->getBpm())
            cachedBpm_.store(*bpm, std::memory_order_relaxed);

    const int numSamples = buffer.getNumSamples();
    if (numSamples <= 0) return;

    MarsDSP::ChronosEngine::Params p{};
    const auto [delL, delR] = computeDelaySamples_();
    p.delaySamplesL = delL;
    p.delaySamplesR = delR;
    p.driveLin = parameters.getRawDriveLin();
    p.mix = parameters.getRawMix();
    p.gainLin = parameters.getRawGainLin();
    p.hpfHz = parameters.getRawHpfHz();
    p.lpfHz = parameters.getRawLpfHz();
    p.filterMode = parameters.getRawFilterMode();
    p.bits = parameters.getRawBits();
    p.adaaOrder = parameters.getADAAOrder();
    p.feedback = parameters.getRawFeedback();
    p.dampHz = parameters.getRawDampHz();
    p.loopCutHz = parameters.getRawLoopCutHz();
    p.crossFeed = parameters.getRawCrossFeed();
    p.loopDrive = parameters.getRawLoopDrive();
    p.loopSatOrder = parameters.getRawLoopSatOrder();
    p.diffusion = parameters.getRawDiffusion();
    p.diffuserSize = parameters.getRawDiffuserSize();
    p.diffModDepth = parameters.getRawDiffModDepth();
    p.diffModRateHz = parameters.getRawDiffModRateHz();
    p.enableDiffuser = parameters.getRawEnableDiffuser();
    p.delaySync = parameters.getRawDelaySync();
    p.delayDivision = parameters.getRawDelayDivision();
    p.delayMode = parameters.getRawDelayMode();
    p.delayModDepth = parameters.getRawDelayModDepth();
    p.delayModRateHz = parameters.getRawDelayModRateHz();
    engine.setParams(p);

    const std::array<float *, 2> io{
        buffer.getWritePointer(0),
        totalNumInputChannels > 1 ? buffer.getWritePointer(1) : nullptr
    };

    float sumInL = 0.0f;
    float sumInR = 0.0f;
    for (int i = 0; i < numSamples; ++i)
    {
        const float l = buffer.getSample(0, i);
        sumInL += l * l;
        if (totalNumInputChannels > 1)
        {
            const float r = buffer.getSample(1, i);
            sumInR += r * r;
        }
    }
    const float rmsL = std::sqrt(sumInL / static_cast<float>(numSamples));
    const float rmsR = (totalNumInputChannels > 1) ? std::sqrt(sumInR / static_cast<float>(numSamples)) : rmsL;

    engine.process(io.data(), totalNumInputChannels, numSamples);

    float sumOutL = 0.0f;
    float sumOutR = 0.0f;
    for (int i = 0; i < numSamples; ++i)
    {
        const float l = buffer.getSample(0, i);
        sumOutL += l * l;
        if (totalNumInputChannels > 1)
        {
            const float r = buffer.getSample(1, i);
            sumOutR += r * r;
        }
    }
    const float wetRmsL = std::sqrt(sumOutL / static_cast<float>(numSamples));
    const float wetRmsR = (totalNumInputChannels > 1) ? std::sqrt(sumOutR / static_cast<float>(numSamples)) : wetRmsL;

    const MarsDSP::GUI::TapFeedFrame frame{
        .rmsL = rmsL,
        .rmsR = rmsR,
        .wetRmsL = wetRmsL,
        .wetRmsR = wetRmsR
    };
    tapFifo_.push(frame);
}

//==============================================================================
bool ChronosProcessor::hasEditor() const
{
    return true;
}

AudioProcessorEditor *ChronosProcessor::createEditor()
{
    return new ChronosEditor(*this);
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
    if (xml == nullptr || !xml->hasTagName(apvts.state.getType())) return;
    ValueTree state(ValueTree::fromXml(*xml));
    if (const int fileVersion = state.getProperty("version"); fileVersion < kStateVersion)
        migrateState_(state, fileVersion);
    state.setProperty("version", kStateVersion, nullptr);
    apvts.replaceState(state);
}

void ChronosProcessor::migrateState_(ValueTree &state, int fromVersion)
{
    // Schema version 5 added delayTimeR and timeLink parameters.
    // Schema version 4 added the delay mode parameter.
    // Schema version 3 added the output filter mode parameter.
    // The default Digital value needs no conversion.
    if (fromVersion < 5)
    {
        float delayTimeVal = 375.0f;
        for (int i = 0; i < state.getNumChildren(); ++i)
        {
            auto child = state.getChild(i);
            if (child.getProperty("id").toString() == "delayTime")
            {
                delayTimeVal = child.getProperty("value");
                break;
            }
        }
        bool hasDelayR = false;
        bool hasTimeLink = false;
        for (int i = 0; i < state.getNumChildren(); ++i)
        {
            auto child = state.getChild(i);
            const String id = child.getProperty("id").toString();
            if (id == "delayTimeR") hasDelayR = true;
            else if (id == "timeLink") hasTimeLink = true;
        }
        if (!hasDelayR)
        {
            ValueTree c("PARAM");
            c.setProperty("id", "delayTimeR", nullptr);
            c.setProperty("value", delayTimeVal, nullptr);
            state.addChild(c, -1, nullptr);
        }
        if (!hasTimeLink)
        {
            ValueTree c("PARAM");
            c.setProperty("id", "timeLink", nullptr);
            c.setProperty("value", 1.0f, nullptr);
            state.addChild(c, -1, nullptr);
        }
    }

    if (fromVersion < 2)
    {
        for (int i = state.getNumChildren() - 1; i >= 0; --i)
        {
            auto child = state.getChild(i);
            const String id = child.getProperty("id").toString();
            if (id == "feedback")
            {
                const float v = child.getProperty("value");
                child.setProperty("value", std::clamp(v, 0.0f, 1.15f), nullptr);
            } else if (id == "drive")
            {
                const float v = child.getProperty("value");
                child.setProperty("value", std::clamp(v, 0.0f, 24.0f), nullptr);
            } else if (id == "diffModDepth")
            {
                // The host can load the state before prepare. Use a safe rate then.
                const double sr = getSampleRate();
                const float safeSr = sr > 0.0 ? static_cast<float>(sr) : 48000.0f;
                const float samples = child.getProperty("value");
                const float ms = samples / safeSr * 1000.0f;
                child.setProperty("value", std::clamp(ms, 0.0f, 1.5f), nullptr);
            } else if (id == "interpolation")
            {
                state.removeChild(i, nullptr);
            }
        }
    }
}

//==============================================================================
AudioProcessor * JUCE_CALLTYPE createPluginFilter()
{
    return new ChronosProcessor();
}
