#include "ChronosProcessor.h"
#include "ChronosEditor.h"
//==============================================================================
ChronosProcessor::ChronosProcessor() : AudioProcessor (BusesProperties()
                       .withInput  ("Input",  AudioChannelSet::stereo(), true)
                       .withOutput ("Output", AudioChannelSet::stereo(), true)
                       )
{
    std::random_device rd;
    std::uniform_int_distribution seedDist { 16386u, UINT32_MAX };

    xorshiftL = seedDist (rd);
    xorshiftR = seedDist (rd);
}

ChronosProcessor::~ChronosProcessor() = default;

AudioProcessorValueTreeState::ParameterLayout ChronosProcessor::createParameterLayout()
{
    AudioProcessorValueTreeState::ParameterLayout layout;
    layout.add(std::make_unique<AudioParameterFloat>(ParameterID {"gain", 1}, "Output Gain", NormalisableRange{-12.0f, 12.0f}, 0.0f));
    layout.add(std::make_unique<AudioParameterInt>(ParameterID {"bits", 1}, "Bit Depth", 1, 32, 24));
    return layout;
}
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

//==============================================================================
void ChronosProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    ignoreUnused (sampleRate, samplesPerBlock);
}

void ChronosProcessor::releaseResources()
{
}

bool ChronosProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    ignoreUnused (layouts);
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

float ChronosProcessor::nextUniform (uint32_t& state) noexcept
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;

    // map the high 24-bits to a float in [0, 1)
    return static_cast<float>(state >> 8) * (1.0f / 8388608.0f);
}

void ChronosProcessor::processBlock (AudioBuffer<float>& buffer,
[[maybe_unused]]MidiBuffer& midiMessages)
{
    ignoreUnused (midiMessages);

    ScopedNoDenormals noDenormals;

    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    // gain
    const float gainDB  = apvts.getRawParameterValue("gain")->load();
    const float gainLin = Decibels::decibelsToGain(gainDB);

    // target bit depth for the output quantizer (1 LSB for a [-1, 1] signal)
    const auto bitDepth = static_cast<int>(apvts.getRawParameterValue("bits")->load());
    const float lsb = std::ldexp(1.0f, 1 - bitDepth);
    const int numSamples = buffer.getNumSamples();

    // final output stage: gain -> TPDF dither -> quantize to target bit depth
    for (auto ch {0uz}; ch < totalNumInputChannels; ++ch)
    {
        auto* data = buffer.getWritePointer(static_cast<int>(ch));
        auto& state = (ch == 0uz) ? xorshiftL : xorshiftR;

        for (auto s {0uz}; s < numSamples; ++s)
        {
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

AudioProcessorEditor* ChronosProcessor::createEditor()
{
    return new GenericAudioProcessorEditor (*this);
}

//==============================================================================
void ChronosProcessor::getStateInformation (MemoryBlock& destData)
{
}

void ChronosProcessor::setStateInformation (const void* data, int sizeInBytes)
{
}
//==============================================================================
AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new ChronosProcessor();
}
