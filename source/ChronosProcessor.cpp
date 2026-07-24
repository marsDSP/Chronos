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

void ChronosProcessor::processBlock (AudioBuffer<float>& buffer,
[[maybe_unused]]MidiBuffer& midiMessages)
{
    ignoreUnused (midiMessages);

    ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    // gain test
    const float gainDB   = apvts.getRawParameterValue("gain")->load();
    const float gainLin  = Decibels::decibelsToGain (gainDB);

    for (auto ch {0uz}; ch < totalNumInputChannels; ++ch)
    {
        auto* data = buffer.getWritePointer(static_cast<int>(ch));
        for (auto smp {0uz}; smp < buffer.getNumSamples(); ++smp)
            data[smp] *= gainLin;
    }

    // advance dither state
    xorshiftL ^= xorshiftL << 13;
    xorshiftL ^= xorshiftL >> 17;
    xorshiftL ^= xorshiftL << 5;

    xorshiftR ^= xorshiftR << 13;
    xorshiftR ^= xorshiftR >> 17;
    xorshiftR ^= xorshiftR << 5;
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
