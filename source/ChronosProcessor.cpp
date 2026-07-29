#include "ChronosProcessor.h"
#include "ChronosEditor.h"
//==============================================================================
ChronosProcessor::ChronosProcessor() : AudioProcessor(BusesProperties()
    .withInput("Input", AudioChannelSet::stereo(), true)
    .withOutput("Output", AudioChannelSet::stereo(), true)
)
{
    std::random_device rd;
    std::uniform_int_distribution seedDist{16386u, UINT32_MAX};

    xorshiftL = seedDist(rd);
    xorshiftR = seedDist(rd);
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

    delayLine.prepare(sampleRate, samplesPerBlock, ChronosParameters::maxDelayTime);
    delayLine.setInterpolation(parameters.getInterpolation());
    delayLine.reset();

    wetBufL_.resize(static_cast<std::size_t>(samplesPerBlock));
    wetBufR_.resize(static_cast<std::size_t>(samplesPerBlock));

    hpf.reset();
    lpf.reset();
    adaa1L_.reset(); adaa1R_.reset();
    adaa2L_.reset(); adaa2R_.reset();

    alignL_.reset();
    alignR_.reset();

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

float ChronosProcessor::nextUniform(uint32_t &state) noexcept
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;

    return static_cast<float>(state >> 8) * (1.0f / 8388608.0f);
}

void ChronosProcessor::processBlock(AudioBuffer<float> &buffer, [[maybe_unused]] MidiBuffer &midiMessages)
{
    ignoreUnused(midiMessages);

    ScopedNoDenormals noDenormals;

    auto totalNumInputChannels = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    if (parameters.getBypass()) return;

    parameters.update();

    const int numSamples = buffer.getNumSamples();
    if (numSamples <= 0) return;

    const double fs = parameters.getSampleRate();
    const double fsSafe = fs > 0.0 ? fs : 48000.0;

    delayLine.setInterpolation(parameters.getInterpolation());

    float* data0 = buffer.getWritePointer(0);
    float* data1 = totalNumInputChannels > 1 ? buffer.getWritePointer(1) : nullptr;

    if (static_cast<int>(wetBufL_.size()) < numSamples)
    {
        wetBufL_.resize(static_cast<std::size_t>(numSamples));
        wetBufR_.resize(static_cast<std::size_t>(numSamples));
    }

    const float delaySamples = parameters.getDelaySamples();
    delayLine.process(data0, data1,
                      wetBufL_.data(),
                      data1 != nullptr ? wetBufR_.data() : nullptr,
                      numSamples, delaySamples, delaySamples);

    hpf.setCoeffForBlock(SVF::SVFType::HighPass, fsSafe, parameters.getHPFFreq(), svfQ, 0.0, numSamples);
    lpf.setCoeffForBlock(SVF::SVFType::LowPass,  fsSafe, parameters.getLPFFreq(), svfQ, 0.0, numSamples);

    const int adaaOrder = parameters.getADAAOrder();
    alignL_.setMode(adaaOrder);
    alignR_.setMode(adaaOrder);

    for (int s = 0; s < numSamples; ++s)
    {
        parameters.smoothen();

        const float driveLin = parameters.getDrive();
        const float mixNorm = parameters.getMix() * 0.01f;
        const float theta = mixNorm * (std::numbers::pi_v<float> * 0.5f);

        const float dryGain = mmCos(theta);
        const float wetGain = mmSin(theta);

        const float dry0 = data0[s];
        const float dry0a = alignL_.processDry(dry0);   // dry delayed kBudget to align with the wet path

        float wet0 = wetBufL_[static_cast<std::size_t>(s)];

        float dry1 = 0.0f;
        float dry1a = 0.0f;
        float wet1 = 0.0f;
        if (data1 != nullptr)
        {
            dry1 = data1[s];
            dry1a = alignR_.processDry(dry1);
            wet1 = wetBufR_[static_cast<std::size_t>(s)];
        }

        float sat0;
        float sat1 = 0.0f;
        switch (adaaOrder)
        {
            case 0:
                sat0 = wet0;
                if (data1 != nullptr) sat1 = wet1;
                break;
            case 1:
                sat0 = static_cast<float>(adaa1L_.process(driveLin * wet0));
                if (data1 != nullptr) sat1 = static_cast<float>(adaa1R_.process(driveLin * wet1));
                break;
            default:
                sat0 = static_cast<float>(adaa2L_.process(driveLin * wet0));
                if (data1 != nullptr) sat1 = static_cast<float>(adaa2R_.process(driveLin * wet1));
                break;
        }

        sat0 = alignL_.processWet(sat0);
        if (data1 != nullptr) sat1 = alignR_.processWet(sat1);

        const M128 wetV = MM(set_ps)(0.0f, 0.0f, sat1, sat0);
        const M128 hpV  = hpf.processBlockStep(wetV);
        const M128 lpV  = lpf.processBlockStep(hpV);
        alignas(16) std::array<float, 4> out;
        MM(storeu_ps)(out.data(), lpV);

        data0[s] = dry0a * dryGain + out[0] * wetGain;
        if (data1 != nullptr) data1[s] = dry1a * dryGain + out[1] * wetGain;

        const float gainLin = parameters.getGain();
        const float lsb = std::ldexp(1.0f, 1 - parameters.getBits());

        for (int ch = 0; ch < totalNumInputChannels; ++ch)
        {
            auto *data = buffer.getWritePointer(static_cast<int>(ch));
            auto &state = ch == 0 ? xorshiftL : xorshiftR;
            const float scaled = data[s] * gainLin;
            const float dither = (nextUniform(state) - nextUniform(state)) * lsb;
            data[s] = std::round((scaled + dither) / lsb) * lsb;
        }
    }
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
