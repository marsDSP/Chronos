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
    return 0.0;
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
    ignoreUnused(samplesPerBlock);
    parameters.prepare(sampleRate);
    parameters.reset();

    dsp::ProcessSpec spec {};
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<uint32>(samplesPerBlock);
    spec.numChannels = 2;

    delayLine.prepare(spec);

    const double numSamples = ChronosParameters::maxDelayTime / 1000.0 * sampleRate;
    const auto maxDelayInSamples = static_cast<int>(std::ceil(numSamples));
    delayLine.setMaximumDelayInSamples(maxDelayInSamples);
    delayLine.reset();

    hpf.reset();
    lpf.reset();
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
    const double fsSafe = (fs > 0.0) ? fs : 48000.0;

    hpf.setCoeffForBlock(SVF::SVFType::HighPass, fsSafe, parameters.getHPFFreq(), svfQ, 0.0, numSamples);
    lpf.setCoeffForBlock(SVF::SVFType::LowPass,  fsSafe, parameters.getLPFFreq(), svfQ, 0.0, numSamples);

    for (auto s {0uz}; s < numSamples; ++s)
    {
        parameters.smoothen();
        delayLine.setDelay(parameters.getDelaySamples());

        const float mixNorm = parameters.getMix() * 0.01f;
        const float theta = mixNorm * (std::numbers::pi_v<float> * 0.5f);
        const float dryGain = std::cos(theta);
        const float wetGain = std::sin(theta);

        float* data0 = buffer.getWritePointer(0);
        const float dry0 = data0[s];
        delayLine.pushSample(0, dry0);
        const float wet0 = delayLine.popSample(0);

        float dry1 = 0.0f;
        float wet1 = 0.0f;
        float* data1 = nullptr;
        if (totalNumInputChannels > 1)
        {
            data1 = buffer.getWritePointer(1);
            dry1 = data1[s];
            delayLine.pushSample(1, dry1);
            wet1 = delayLine.popSample(1);
        }

        const M128 wetV = MM(set_ps)(0.0f, 0.0f, wet1, wet0);   // lane0=L, lane1=R
        const M128 hpV  = hpf.processBlockStep(wetV);
        const M128 lpV  = lpf.processBlockStep(hpV);
        alignas(16) float out[4];
        MM(storeu_ps)(out, lpV);

        data0[s] = dry0 * dryGain + out[0] * wetGain;
        if (totalNumInputChannels > 1) data1[s] = dry1 * dryGain + out[1] * wetGain;

        const float gainLin = parameters.getGain();
        const float lsb = std::ldexp(1.0f, 1 - parameters.getBits());

        for (auto ch {0uz}; ch < totalNumInputChannels; ++ch)
        {
            auto *data = buffer.getWritePointer(static_cast<int>(ch));
            auto &state = ch == 0uz ? xorshiftL : xorshiftR;

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
