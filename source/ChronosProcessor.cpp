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

    for (auto& f : hpf) f.reset();
    for (auto& f : lpf) f.reset();
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

    // map the high 24-bits to a float in [0, 1)
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

    constexpr double pi = std::numbers::pi_v<double>;
    const double fs = parameters.getSampleRate();
    const double fsSafe = (fs > 0.0) ? fs : 48000.0;
    const double nyq = 0.49 * fsSafe;

    for (auto s {0uz}; s < numSamples; ++s)
    {
        parameters.smoothen();
        delayLine.setDelay(parameters.getDelaySamples());

        const float mixNorm = parameters.getMix() * 0.01f;
        const float theta = mixNorm * (std::numbers::pi_v<float> * 0.5f);
        const float dryGain = std::cos(theta);
        const float wetGain = std::sin(theta);

        const double hpfF = std::clamp(static_cast<double>(parameters.getHPFFreq()), 10.0, nyq);
        const double lpfF = std::clamp(static_cast<double>(parameters.getLPFFreq()), 10.0, nyq);
        alignas(16) const float angles[4] = {
            static_cast<float>(pi * hpfF / fsSafe),  // ch0 HPF
            static_cast<float>(pi * lpfF / fsSafe),  // ch0 LPF
            static_cast<float>(pi * hpfF / fsSafe),  // ch1 HPF
            static_cast<float>(pi * lpfF / fsSafe),  // ch1 LPF
        };
        alignas(16) float gt[4];
        MM(storeu_ps)(gt, mmTan(MM(loadu_ps)(angles)));

        for (auto ch {0uz}; ch < totalNumInputChannels; ++ch)
        {
            auto *data = buffer.getWritePointer(static_cast<int>(ch));
            const float dry = data[s];
            delayLine.pushSample(static_cast<int>(ch), dry);
            float wet = delayLine.popSample(static_cast<int>(ch));
            const std::size_t base = 2 * ch;
            hpf[ch].setParamsFromG(SVF::SVFType::HighPass, svfQ, 0.0, gt[base + 0]);
            wet = hpf[ch].processSample(wet);
            lpf[ch].setParamsFromG(SVF::SVFType::LowPass,  svfQ, 0.0, gt[base + 1]);
            wet = lpf[ch].processSample(wet);
            data[s] = dry * dryGain + wet * wetGain;
        }

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
