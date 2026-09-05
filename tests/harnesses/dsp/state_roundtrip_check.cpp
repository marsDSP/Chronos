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

// The tag of the editor state side tree on a serialised root.
constexpr const char* kEditorTag = "EDITOR";

// Serialise the APVTS state with the schema version stamped in and
// the editor side tree appended, mirroring the processor path.
MemoryBlock saveState (AudioProcessorValueTreeState& a, const ValueTree& editorState)
{
    ValueTree s = a.copyState();
    s.setProperty ("version", 5, nullptr);
    s.appendChild (editorState.createCopy(), nullptr);
    MemoryBlock block;
    AudioProcessor::copyXmlToBinary (*s.createXml(), block);
    return block;
}

// Load a state block into the APVTS. Mirrors the processor path: extract
// and remove any EDITOR child before the version stamp and the replace.
void loadState (AudioProcessorValueTreeState& a, const MemoryBlock& block, ValueTree& editorState)
{
    auto xml = AudioProcessor::getXmlFromBinary (block.getData(), (int) block.getSize());
    if (xml == nullptr || ! xml->hasTagName (a.state.getType()))
        return;
    ValueTree s (ValueTree::fromXml (*xml));

    const auto editor = s.getChildWithName (kEditorTag);
    if (editor.isValid())
    {
        s.removeChild (editor, nullptr);
        editorState = editor;
    }

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
        ValueTree editorSide { kEditorTag };

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

        const MemoryBlock save1 = saveState (a, editorSide);
        loadState (a, save1, editorSide);
        const MemoryBlock save2 = saveState (a, editorSide);

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

    // ----------------------------------------------------------------
    // Editor side tree: the width and the tab indices ride on the
    // serialised root under the EDITOR tag, never on the parameter
    // tree. A version-5 file without the child loads unchanged.
    // ----------------------------------------------------------------
    g_section = "editor-side-tree";
    {
        StubProcessor proc;
        auto& a = proc.apvts;
        ValueTree side { kEditorTag };

        setDenorm (a, "delayTime", 1234.0f);

        side.setProperty ("editorWidth", 1600, nullptr);
        side.setProperty ("timeTab", 1, nullptr);
        side.setProperty ("characterTab", 1, nullptr);

        const MemoryBlock save1 = saveState (a, side);

        // Perturb the side tree, then reload. The saved values return.
        side.setProperty ("editorWidth", 800, nullptr);
        side.setProperty ("timeTab", 0, nullptr);
        loadState (a, save1, side);

        CHECK (static_cast<int> (side.getProperty ("editorWidth")) == 1600);
        CHECK (static_cast<int> (side.getProperty ("timeTab")) == 1);
        CHECK (static_cast<int> (side.getProperty ("characterTab")) == 1);

        // The parameter tree carries no EDITOR child and no width
        // property at any point in the session.
        CHECK (! a.state.getChildWithName (kEditorTag).isValid());
        CHECK (! a.state.hasProperty ("editorWidth"));

        // The serialised session carries the EDITOR child once.
        const auto again = saveState (a, side);
        auto xml = AudioProcessor::getXmlFromBinary (again.getData(), (int) again.getSize());
        CHECK (xml != nullptr);
        int editorChildren = 0;
        for (int i = 0; i < xml->getNumChildElements(); ++i)
            if (auto* el = xml->getChildElement (i); el != nullptr && el->hasTagName (kEditorTag))
                ++editorChildren;
        CHECK (editorChildren == 1);

        // A version-5 session file without the child loads unchanged:
        // the side tree keeps its current values.
        ValueTree bareSide { kEditorTag };
        bareSide.setProperty ("editorWidth", 900, nullptr);
        const MemoryBlock legacy = [&]
        {
            ValueTree s = a.copyState();
            s.setProperty ("version", 5, nullptr);
            MemoryBlock block;
            AudioProcessor::copyXmlToBinary (*s.createXml(), block);
            return block;
        }();
        loadState (a, legacy, bareSide);
        CHECK (static_cast<int> (bareSide.getProperty ("editorWidth")) == 900);
        CHECK (getDenorm (a, "delayTime") >= 1233.0f && getDenorm (a, "delayTime") <= 1235.0f);
    }

    std::println("=== state_roundtrip_check OK ===");
    return 0;
}
