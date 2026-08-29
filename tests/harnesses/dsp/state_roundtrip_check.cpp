// tests/harnesses/dsp/state_roundtrip_check.cpp
//
// State schema harness. Verifies the processor state round-trips byte for
// byte, and that a version-1 fixture loads with values clamped to the legal
// range. The harness builds a real APVTS from ChronosParameters and mirrors
// the processor getStateInformation / setStateInformation path. It links the
// JUCE processor modules with SharedCode, and needs no JucePlugin macros.

#include <juce_audio_processors/juce_audio_processors.h>

using namespace juce;

#include "ChronosParameters.h"

#include <cstdint>
#include <cstdio>
#include <print>
#include <cstring>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond)                                                            \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...)                                                         \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

// Minimal AudioProcessor that owns a real APVTS with the Chronos layout.
class StubProcessor final : public AudioProcessor
{
public:
    StubProcessor()
        : apvts (*this, nullptr, "Parameters", ChronosParameters::createParameterLayout()) {}

    const String getName() const override { return {}; }
    void prepareToPlay (double, int) override {}
    void releaseResources() override {}
    void processBlock (AudioBuffer<float>&, MidiBuffer&) override {}
    using AudioProcessor::processBlock;
    double getTailLengthSeconds() const override { return {}; }
    bool acceptsMidi() const override { return {}; }
    bool producesMidi() const override { return {}; }
    bool isMidiEffect() const override { return false; }
    AudioProcessorEditor* createEditor() override { return {}; }
    bool hasEditor() const override { return false; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const String getProgramName (int) override { return {}; }
    void changeProgramName (int, const String&) override {}
    void getStateInformation (MemoryBlock&) override {}
    void setStateInformation (const void*, int) override {}

    AudioProcessorValueTreeState apvts;
};

// Set a denormalised value on a parameter through the APVTS.
void setDenorm (AudioProcessorValueTreeState& a, const char* id, float denorm)
{
    auto* p = a.getParameter (id);
    CHECK (p != nullptr);
    p->setValueNotifyingHost (p->getNormalisableRange().convertTo0to1 (denorm));
}

// Read the denormalised value of a parameter.
float getDenorm (const AudioProcessorValueTreeState& a, const char* id)
{
    auto* raw = a.getRawParameterValue (id);
    CHECK (raw != nullptr);
    return raw->load();
}

// Serialise the APVTS state with the schema version stamped in.
MemoryBlock saveState (AudioProcessorValueTreeState& a)
{
    ValueTree s = a.copyState();
    s.setProperty ("version", 5, nullptr);
    MemoryBlock block;
    AudioProcessor::copyXmlToBinary (*s.createXml(), block);
    return block;
}

// Load a state block into the APVTS. Mirrors the processor path: read the
// version, then stamp the current version, then replace the state.
void loadState (AudioProcessorValueTreeState& a, const MemoryBlock& block)
{
    auto xml = AudioProcessor::getXmlFromBinary (block.getData(), (int) block.getSize());
    if (xml == nullptr || ! xml->hasTagName (a.state.getType()))
        return;
    ValueTree s (ValueTree::fromXml (*xml));
    s.setProperty ("version", 5, nullptr);
    a.replaceState (s);
}

} // namespace

int main()
{
    ScopedJuceInitialiser_GUI gui;

    // Round-trip: save, load, save produces byte-identical output.
    g_section = "round-trip";
    {
        StubProcessor proc;
        auto& a = proc.apvts;

        // Move several parameters off their defaults.
        setDenorm (a, "delayTime", 1234.0f);
        setDenorm (a, "delayTimeR", 2345.0f);
        setDenorm (a, "timeLink", 0.0f);
        setDenorm (a, "feedback", 0.55f);
        setDenorm (a, "mix", 42.0f);
        setDenorm (a, "drive", 9.0f);
        setDenorm (a, "dampHz", 3200.0f);
        setDenorm (a, "enableDiffuser", 1.0f);
        setDenorm (a, "filterMode", 1.0f);
        setDenorm (a, "delayMode", 1.0f);

        const MemoryBlock save1 = saveState (a);
        loadState (a, save1);
        const MemoryBlock save2 = saveState (a);

        CHECK (save1.getSize() > 0);
        CHECK (save1.getSize() == save2.getSize());
        if (std::memcmp (save1.getData(), save2.getData(), save1.getSize()) != 0)
            FAIL("save1 != save2 after a round-trip ({} bytes)",
                  static_cast<unsigned long long> (save1.getSize()));

        // Analog mode and BBD delay mode survive the round-trip
        const float mode = getDenorm (a, "filterMode");
        CHECK (mode == 1.0f);
        const float dMode = getDenorm (a, "delayMode");
        CHECK (dMode == 1.0f);
    }

    // Version-1 fixture: no version property, one out-of-range value.
    g_section = "version-1 fixture";
    {
        StubProcessor proc;
        auto& a = proc.apvts;

        const String fixtureXml =
            String ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
            + "<Parameters>"
            + "<PARAM id=\"delayTime\" value=\"900.0\"/>"
            + "<PARAM id=\"feedback\" value=\"5.0\"/>"
            + "<PARAM id=\"mix\" value=\"30.0\"/>"
            + "<PARAM id=\"drive\" value=\"6.0\"/>"
            + "<PARAM id=\"enableDiffuser\" value=\"1.0\"/>"
            + "</Parameters>";

        auto xml = parseXML (fixtureXml);
        CHECK (xml != nullptr);
        CHECK (xml->hasTagName ("Parameters"));
        ValueTree s (ValueTree::fromXml (*xml));
        // Absent version reads as zero, which means a version-1 state.
        CHECK (! s.hasProperty ("version"));
        s.setProperty ("version", 5, nullptr);
        a.replaceState (s);

        // The out-of-range feedback must clamp to the legal maximum.
        const float fb = getDenorm(a, "feedback");
        CHECK(fb > 1.14f && fb < 1.16f);
        const float delay = getDenorm(a, "delayTime");
        CHECK(delay >= 1.0f && delay <= 5000.001f);
        // Stored tree without filterMode loads and reports Digital (0)
        const float mode = getDenorm(a, "filterMode");
        CHECK(mode == 0.0f);
        // Stored tree without delayMode loads and reports Digital (0)
        const float dMode = getDenorm(a, "delayMode");
        CHECK(dMode == 0.0f);
        // The live state now carries the current schema version.
        CHECK (static_cast<int> (a.state.getProperty ("version")) == 5);
    }

    std::println("=== state_roundtrip_check OK ===");
    return 0;
}
