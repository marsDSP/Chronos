#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
PluginProcessor::PluginProcessor() : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
                            params(vts)
{
}
//==============================================================================
const juce::String PluginProcessor::getName() const
{
    return JucePlugin_Name;
}

bool PluginProcessor::acceptsMidi() const
{
#if JucePlugin_WantsMidiInput
    return true;
#else
    return false;
#endif
}

bool PluginProcessor::producesMidi() const
{
#if JucePlugin_ProducesMidiOutput
    return true;
#else
    return false;
#endif
}

bool PluginProcessor::isMidiEffect() const
{
#if JucePlugin_IsMidiEffect
    return true;
#else
    return false;
#endif
}

double PluginProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int PluginProcessor::getNumPrograms()
{
    return 1; // NB: some hosts don't cope very well if you tell them there are 0 programs,
              // so this should be at least 1, even if you're not really implementing programs.
}

int PluginProcessor::getCurrentProgram()
{
    return 0;
}

void PluginProcessor::setCurrentProgram(int index)
{
    juce::ignoreUnused(index);
}

const juce::String PluginProcessor::getProgramName(int index)
{
    juce::ignoreUnused(index);
    return {};
}

void PluginProcessor::changeProgramName(int index, const juce::String &newName)
{
    juce::ignoreUnused(index, newName);
}

void PluginProcessor::parameterChanged(const juce::String& paramID, float newValue)
{
}

void PluginProcessor::updateParameters()
{
    const auto oversampling_choice = params.oversample->getIndex();
    if (oversampling_choice >= 0 && static_cast<size_t>(oversampling_choice) < processDSP.size())
    {processDSP[static_cast<size_t>(oversampling_choice)].updateParams(params);}
}

void PluginProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    for (int i {}; i < processDSP.size(); ++i)
    {
        processDSP[i].prepareDSP(sampleRate, samplesPerBlock, 2, i, params);
    }
}

void PluginProcessor::releaseResources()
{
}

bool PluginProcessor::isBusesLayoutSupported(const BusesLayout &layouts) const
{
#if JucePlugin_IsMidiEffect
    juce::ignoreUnused(layouts);
    return true;
#else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono() &&
        layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

        // This checks if the input layout matches the output layout
#if !JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
#endif

    return true;
#endif
}
void PluginProcessor::processBlock(juce::AudioBuffer<float> &buffer,
                                   juce::MidiBuffer &midiMessages)
{
    juce::ignoreUnused(midiMessages);

    updateParameters();

    // Ensure the active processing path receives the latest parameter targets
    // so that its internal smoother object can drive per-sample values.

    const auto oversampleChoice = params.oversample->getIndex();
    if (oversampleChoice >= 0 && static_cast<size_t>(oversampleChoice) < processDSP.size())
    {
        processDSP[static_cast<size_t>(oversampleChoice)].updateParams(params);
    }

    juce::ScopedNoDenormals noDenormals;

    if (oversampleChoice >= 0 && static_cast<size_t>(oversampleChoice) < processDSP.size())
        processDSP[static_cast<size_t>(oversampleChoice)].processDSP(buffer, buffer.getNumSamples());

}

//==============================================================================
bool PluginProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor *PluginProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor(*this);
}

//==============================================================================
void PluginProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    copyXmlToBinary( *vts.copyState().createXml(), destData );
}

void PluginProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // Restore the parameters state from the given data
    const std::unique_ptr xml(getXmlFromBinary( data, sizeInBytes ));
    if ( xml.get() != nullptr && xml->hasTagName( vts.state.getType() ))
    {
        vts.replaceState(juce::ValueTree::fromXml( *xml ));
    }
}
//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor *JUCE_CALLTYPE createPluginFilter()
{
    return new PluginProcessor();
}
