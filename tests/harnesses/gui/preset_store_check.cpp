// tests/harnesses/gui/preset_store_check.cpp
//
// Preset store and manager harness. Verifies the file layer
// and the policy layer against a real APVTS built from the
// Chronos parameter layout. Links the JUCE processor modules
// with SharedCode, and needs no JucePlugin macros.

#include <juce_audio_processors/juce_audio_processors.h>

using namespace juce;

#include "ChronosParameters.h"
#include "presets/PresetStore.h"
#include "presets/PresetManager.h"

using namespace MarsDSP::Presets;

#include <cstdint>
#include <cstdio>
#include <print>
#include <cstring>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond)                                                            \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...)                                                         \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

// Minimal AudioProcessor that owns a real APVTS with the Chronos layout.
// Mirrors the processor state path so the preset manager loads through
// the same deserialiser the host uses.
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

    // Mirror the processor state path, including the editor side tree.
    void getStateInformation (MemoryBlock& destData) override
    {
        ValueTree state = apvts.copyState();
        state.setProperty ("version", kStateVersion, nullptr);
        state.appendChild (editorSide.createCopy(), nullptr);
        copyXmlToBinary (*state.createXml(), destData);
    }

    void setStateInformation (const void* data, int sizeInBytes) override
    {
        auto xml = AudioProcessor::getXmlFromBinary (data, sizeInBytes);
        if (xml == nullptr || ! xml->hasTagName (apvts.state.getType())) return;
        ValueTree state (ValueTree::fromXml (*xml));

        const auto editor = state.getChildWithName ("EDITOR");
        if (editor.isValid())
        {
            state.removeChild (editor, nullptr);
            editorSide = editor;
        }

        if (const int fileVersion = state.getProperty ("version"); fileVersion < kStateVersion)
            migrateState_ (state, fileVersion);
        state.setProperty ("version", kStateVersion, nullptr);
        apvts.replaceState (state);
    }

    AudioProcessorValueTreeState apvts;

    // The editor side tree, as the processor owns one.
    ValueTree editorSide { "EDITOR" };

private:
    static constexpr int kStateVersion = 5;

    // Copy of the processor migration path so a preset file
    // loads through the same code as a host session recall.
    void migrateState_ (ValueTree& state, int fromVersion)
    {
        if (fromVersion < 5)
        {
            float delayTimeVal = 375.0f;
            for (int i = 0; i < state.getNumChildren(); ++i)
            {
                auto child = state.getChild (i);
                if (child.getProperty ("id").toString() == "delayTime")
                {
                    delayTimeVal = child.getProperty ("value");
                    break;
                }
            }
            bool hasDelayR = false, hasTimeLink = false;
            for (int i = 0; i < state.getNumChildren(); ++i)
            {
                auto child = state.getChild (i);
                const String id = child.getProperty ("id").toString();
                if (id == "delayTimeR") hasDelayR = true;
                else if (id == "timeLink") hasTimeLink = true;
            }
            if (! hasDelayR)
            {
                ValueTree c ("PARAM");
                c.setProperty ("id", "delayTimeR", nullptr);
                c.setProperty ("value", delayTimeVal, nullptr);
                state.addChild (c, -1, nullptr);
            }
            if (! hasTimeLink)
            {
                ValueTree c ("PARAM");
                c.setProperty ("id", "timeLink", nullptr);
                c.setProperty ("value", 1.0f, nullptr);
                state.addChild (c, -1, nullptr);
            }
        }

        if (fromVersion < 2)
        {
            for (int i = state.getNumChildren() - 1; i >= 0; --i)
            {
                auto child = state.getChild (i);
                const String id = child.getProperty ("id").toString();
                if (id == "feedback")
                    child.setProperty ("value", std::clamp (static_cast<float> (child.getProperty ("value")), 0.0f, 1.15f), nullptr);
                else if (id == "drive")
                    child.setProperty ("value", std::clamp (static_cast<float> (child.getProperty ("value")), 0.0f, 24.0f), nullptr);
                else if (id == "diffModDepth")
                    child.setProperty ("value", std::clamp (static_cast<float> (child.getProperty ("value")), 0.0f, 1.5f), nullptr);
                else if (id == "interpolation")
                    state.removeChild (i, nullptr);
            }
        }
    }
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

// The 27 preset parameter IDs. Bypass is not preset state, so the
// round trip excludes it: a saved file carries no bypass child, and
// a load leaves the live bypass value alone.
static const char* const kParamIDs[] = {
    "gain",          "bits",          "delayTime",     "delayTimeR",
    "timeLink",      "delaySync",     "delayDivision",  "delayMode",
    "filterMode",    "hpfFreq",       "lpfFreq",
    "mix",           "drive",         "adaaOrder",      "feedback",
    "dampHz",        "loopCutHz",      "crossFeed",      "loopDrive",
    "loopSatOrder",   "delayModDepth",  "delayModRateHz", "enableDiffuser",
    "diffusion",      "diffuserSize",   "diffModDepth",   "diffModRateHz"
};
constexpr int kNumParams = static_cast<int> (std::size (kParamIDs));

// Return a pseudo-random value for one parameter. The seed fixes the value per run.
float randomDenorm (AudioProcessorValueTreeState& a, const char* id, int seed)
{
    auto* p = a.getParameter (id);
    if (p == nullptr) return 0.0f;
    const auto range = p->getNormalisableRange();
    return range.convertFrom0to1 (static_cast<float> ((seed * 2654435761u) % 1000) / 999.0f);
}

// Build a version-N state tree XML string for the migration test.
// Every preset parameter is present at its default, so the file
// passes the total recall check. outOfRange pushes one value outside
// its range. unknownParam adds a parameter that does not exist.
String buildVersionedXml (int version, const String& rootTag, bool outOfRange, bool unknownParam)
{
    StubProcessor probe;
    String xml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<" + rootTag;
    if (version > 0)
        xml += " version=\"" + String (version) + "\"";
    xml += ">";

    for (int i = 0; i < kNumParams; ++i)
    {
        float denorm = 0.5f;
        if (auto* p = probe.apvts.getParameter (kParamIDs[i]))
            denorm = p->getNormalisableRange().convertFrom0to1 (p->getDefaultValue());
        String valueText (denorm, 6);
        if (outOfRange && String (kParamIDs[i]) == "feedback")
            valueText = "5.0";
        xml += "<PARAM id=\"" + String (kParamIDs[i]) + "\" value=\"" + valueText + "\"/>";
    }

    if (unknownParam)
        xml += "<PARAM id=\"doesNotExist\" value=\"1.0\"/>";
    xml += "</" + rootTag + ">";
    return xml;
}

} // namespace

int main()
{
    ScopedJuceInitialiser_GUI gui;

    // Use a temporary preset directory so the harness never touches the
    // real user preset folder.
    const auto tempRoot = File::getSpecialLocation (File::tempDirectory)
        .getChildFile ("chronos_preset_store_check");
    tempRoot.deleteRecursively();
    tempRoot.createDirectory();
    PresetStore store;
    store.setRootDirectory (tempRoot);

    // ----------------------------------------------------------------
    // 1. Round trip: set all 28 parameters, save, perturb, then load.
    //    Compare every value to a tolerance. Repeat 200 times.
    // ----------------------------------------------------------------
    g_section = "round-trip";
    {
        StubProcessor proc;
        PresetManager pm (proc, proc.apvts);
        pm.getStore().setRootDirectory (tempRoot);

        // JUCE writes float32 values through text with reduced precision.
        // A save and a load can shift a value by about 1e-7 relative.
        // Allow a relative tolerance with an absolute floor for near zero.
        constexpr float kRoundTripRel = 1e-5f;
        constexpr float kRoundTripAbs = 1e-6f;

        for (int iter = 0; iter < 200; ++iter)
        {
            const String name = "RoundTrip" + String (iter);

            // Set a random value for each parameter. Read back the stored
            // value so the discrete parameters compare equal after the load.
            std::vector<float> saved (std::size (kParamIDs));
            for (int i = 0; i < kNumParams; ++i)
            {
                setDenorm (proc.apvts, kParamIDs[i],
                           randomDenorm (proc.apvts, kParamIDs[i], i + 1 + iter * 31));
                saved[std::size_t (i)] = getDenorm (proc.apvts, kParamIDs[i]);
            }

            CHECK (pm.saveAs (name, "harness", "test"));
            CHECK (pm.getCurrentName() == name);
            CHECK (! pm.isModified());

            // Perturb every parameter so the load must restore the saved values.
            for (int i = 0; i < kNumParams; ++i)
                setDenorm (proc.apvts, kParamIDs[i],
                           randomDenorm (proc.apvts, kParamIDs[i], i + 100 + iter * 31));

            CHECK (pm.isModified());

            const auto file = store.presetFile ({}, name);
            CHECK (pm.loadPreset (file));

            // replaceState fires the parameter listeners synchronously, so
            // the modified flag is true right after loadPreset returns. The
            // async clear is queued; clear manually for the headless harness.
            pm.clearModified();

            for (int i = 0; i < kNumParams; ++i)
            {
                const auto v = getDenorm (proc.apvts, kParamIDs[i]);
                const float diff = std::fabs (v - saved[std::size_t (i)]);
                const float limit = kRoundTripRel * std::fabs (saved[std::size_t (i)]) + kRoundTripAbs;
                if (diff > limit)
                    FAIL("iter {} param {} did not round-trip: got {} expected {} (diff {})",
                         iter, kParamIDs[i], v, saved[std::size_t (i)], diff);
            }

            CHECK (! pm.isModified());
            CHECK (pm.getCurrentName() == name);
        }
    }

    // ----------------------------------------------------------------
    // 2. Metadata: presetName, author, category survive a round trip.
    //    A file missing all three still loads.
    // ----------------------------------------------------------------
    g_section = "metadata";
    {
        StubProcessor proc;
        PresetManager pm (proc, proc.apvts);
        pm.getStore().setRootDirectory (tempRoot);

        setDenorm (proc.apvts, "delayTime", 200.0f);
        CHECK (pm.saveAs ("Meta", "author", "cat"));
        CHECK (pm.getCurrentName() == "Meta");

        const auto file = store.presetFile ({}, "Meta");
        CHECK (pm.loadPreset (file));
        pm.clearModified();

        CHECK (pm.getCurrentName() == "Meta");

        // A file missing all three metadata properties still loads.
        // Write a bare state tree through the processor save path.
        {
            MemoryBlock block;
            proc.getStateInformation (block);
            const auto xml = AudioProcessor::getXmlFromBinary (block.getData(), (int) block.getSize());
            CHECK (xml != nullptr);
            const auto bareFile = tempRoot.getChildFile ("Bare.chronos");
            CHECK (PresetStore::savePresetFile (bareFile, xml->toString()));

            // Snapshot a parameter before load.
            setDenorm (proc.apvts, "mix", 10.0f);
            const float before = getDenorm (proc.apvts, "mix");

            CHECK (pm.loadPreset (bareFile));
            pm.clearModified();

            // The mix parameter restores to the value in the bare file.
            const float after = getDenorm (proc.apvts, "mix");
            CHECK (std::fabs (after - 35.0f) < 1.0f);
        }
    }

    // ----------------------------------------------------------------
    // 3. Migration: a v1 through v4 preset file loads through the same
    //    path as a host session recall of the same tree.
    // ----------------------------------------------------------------
    g_section = "migration";
    {
        const String rootTag = StubProcessor().apvts.state.getType().toString();

        for (int version = 1; version <= 4; ++version)
        {
            StubProcessor procA;
            StubProcessor procB;

            const auto xmlText = buildVersionedXml (version, rootTag, false, false);
            const auto file = tempRoot.getChildFile ("v" + String (version) + ".chronos");
            CHECK (PresetStore::savePresetFile (file, xmlText));

            // Load the preset through the manager (procA) and through the
            // processor state path directly (procB). Both must match.
            PresetManager pm (procA, procA.apvts);
            pm.getStore().setRootDirectory (tempRoot);
            CHECK (pm.loadPreset (file));
            pm.clearModified();

            MemoryBlock blob;
            auto xml = parseXML (xmlText);
            CHECK (xml != nullptr);
            AudioProcessor::copyXmlToBinary (*xml, blob);
            procB.setStateInformation (blob.getData(), (int) blob.getSize());

            for (int i = 0; i < kNumParams; ++i)
            {
                const auto a = getDenorm (procA.apvts, kParamIDs[i]);
                const auto b = getDenorm (procB.apvts, kParamIDs[i]);
                if (std::fabs (a - b) > 1e-6f)
                    FAIL("v{} param {} mismatch: manager {} vs direct {}",
                         version, kParamIDs[i], a, b);
            }
        }
    }

    // ----------------------------------------------------------------
    // 4. Hostile files: each fails cleanly and leaves the state unchanged.
    // ----------------------------------------------------------------
    g_section = "hostile-files";
    {
        auto testHostile = [&] (const String& name, const String& content)
        {
            StubProcessor proc;
            PresetManager pm (proc, proc.apvts);
            pm.getStore().setRootDirectory (tempRoot);

            // Set a known state so we can detect a change.
            setDenorm (proc.apvts, "delayTime", 500.0f);
            const float before = getDenorm (proc.apvts, "delayTime");
            const String beforeName = pm.getCurrentName();

            const auto file = tempRoot.getChildFile (name + ".chronos");
            file.replaceWithText (content);

            CHECK (! pm.loadPreset (file));

            // The state must be untouched.
            const float after = getDenorm (proc.apvts, "delayTime");
            CHECK (std::fabs (after - before) < 1e-6f);
            CHECK (pm.getCurrentName() == beforeName);
        };

        testHostile ("empty", "");
        testHostile ("truncated", "<?xml version=\"1.0\"?><Parameters><PARAM id=\"delayTi");
        testHostile ("wrongtag", "<?xml version=\"1.0\"?><WrongRoot/>");

        // A parameter outside its legal range fails the load.
        // The state stays unchanged.
        {
            StubProcessor proc;
            PresetManager pm (proc, proc.apvts);
            pm.getStore().setRootDirectory (tempRoot);
            setDenorm (proc.apvts, "delayTime", 500.0f);
            const float before = getDenorm (proc.apvts, "delayTime");
            const auto xmlText = buildVersionedXml (1, StubProcessor().apvts.state.getType().toString(), true, false);
            const auto file = tempRoot.getChildFile ("outofrange.chronos");
            file.replaceWithText (xmlText);
            CHECK (! pm.loadPreset (file));
            const float after = getDenorm (proc.apvts, "delayTime");
            CHECK (std::fabs (after - before) < 1e-6f);
        }

        // A parameter that no longer exists fails the load.
        // The state stays unchanged.
        {
            StubProcessor proc;
            PresetManager pm (proc, proc.apvts);
            pm.getStore().setRootDirectory (tempRoot);
            setDenorm (proc.apvts, "delayTime", 500.0f);
            const float before = getDenorm (proc.apvts, "delayTime");
            const auto xmlText = buildVersionedXml (1, StubProcessor().apvts.state.getType().toString(), false, true);
            const auto file = tempRoot.getChildFile ("unknownparam.chronos");
            file.replaceWithText (xmlText);
            CHECK (! pm.loadPreset (file));
            const float after = getDenorm (proc.apvts, "delayTime");
            CHECK (std::fabs (after - before) < 1e-6f);
        }

        // A missing directory: enumeration returns empty, load fails.
        {
            StubProcessor proc;
            PresetManager pm (proc, proc.apvts);
            pm.getStore().setRootDirectory (tempRoot.getChildFile ("missing"));
            CHECK (pm.getUserPresets().empty());
            // A load of a nonexistent file fails cleanly.
            CHECK (! pm.loadPreset (tempRoot.getChildFile ("nope.chronos")));
        }
    }

    // ----------------------------------------------------------------
    // 5. Naming: path separators, leading dots, and an empty name
    //    are rejected. A duplicate name does not overwrite silently.
    // ----------------------------------------------------------------
    g_section = "naming";
    {
        StubProcessor proc;
        PresetManager pm (proc, proc.apvts);
        pm.getStore().setRootDirectory (tempRoot);

        // Path separators and leading dots are stripped, not rejected by
        // the sanitiser. An empty result after sanitising is rejected
        // by the caller (the manager checks for an empty name).
        CHECK (PresetStore::sanitiseName ("foo/bar") == "foo_bar");
        CHECK (PresetStore::sanitiseName (".foo") == "foo");
        CHECK (PresetStore::sanitiseName ("   ").trim().isEmpty());

        // An empty name does not save.
        CHECK (! pm.saveAs ("", "a", "c"));

        // A duplicate name does not overwrite the existing file.
        CHECK (pm.saveAs ("Dup", "a", "c"));
        const auto dupFile = store.presetFile ({}, "Dup");
        const auto dupContent = dupFile.loadFileAsString();

        // Save over the same name through the store refuses.
        CHECK (! PresetStore::savePresetFile (dupFile, "<Parameters/>"));

        // The original content is intact.
        CHECK (dupFile.loadFileAsString() == dupContent);

        // Rename refuses to clobber an existing file.
        CHECK (pm.saveAs ("Target", "a", "c"));
        CHECK (! pm.renameCurrent ("Dup"));
    }

    // ----------------------------------------------------------------
    // 6. Bypass exclusion: no saved file carries a bypass child, and a
    //    load never changes the live bypass value. An older file with a
    //    bypass child loads, and the child is ignored.
    // ----------------------------------------------------------------
    g_section = "bypass-exclusion";
    {
        StubProcessor proc;
        PresetManager pm (proc, proc.apvts);
        pm.getStore().setRootDirectory (tempRoot);

        // Save while bypassed. The file carries no bypass child.
        setDenorm (proc.apvts, "bypass", 1.0f);
        setDenorm (proc.apvts, "delayTime", 250.0f);
        CHECK (pm.saveAs ("Bypassed", "a", "c"));

        const auto file = store.presetFile ({}, "Bypassed");
        const auto saved = parseXML (file.loadFileAsString());
        CHECK (saved != nullptr);
        for (int i = 0; i < saved->getNumChildElements(); ++i)
        {
            auto* el = saved->getChildElement (i);
            if (el != nullptr && el->hasTagName ("PARAM"))
                CHECK (el->getStringAttribute ("id") != "bypass");
        }

        // Unbypass and perturb. The load must not re-bypass.
        setDenorm (proc.apvts, "bypass", 0.0f);
        for (int i = 0; i < kNumParams; ++i)
            setDenorm (proc.apvts, kParamIDs[i],
                       randomDenorm (proc.apvts, kParamIDs[i], i + 7));
        CHECK (pm.loadPreset (file));
        pm.clearModified();
        CHECK (getDenorm (proc.apvts, "bypass") == 0.0f);
        CHECK (std::fabs (getDenorm (proc.apvts, "delayTime") - 250.0f) < 0.01f);

        // A hand-edited file that carries a bypass child still loads.
        // The child is ignored, and the live bypass survives.
        auto patched = parseXML (file.loadFileAsString());
        CHECK (patched != nullptr);
        auto* bypassChild = new XmlElement ("PARAM");
        bypassChild->setAttribute ("id", "bypass");
        bypassChild->setAttribute ("value", 1.0f);
        patched->addChildElement (bypassChild);

        const auto legacy = tempRoot.getChildFile ("LegacyBypass.chronos");
        CHECK (PresetStore::savePresetFile (legacy, patched->toString()));
        setDenorm (proc.apvts, "bypass", 0.0f);
        CHECK (pm.loadPreset (legacy));
        pm.clearModified();
        CHECK (getDenorm (proc.apvts, "bypass") == 0.0f);
    }

    // ----------------------------------------------------------------
    // 7. Bank identity: a file in a bank reports that bank, a root
    //    file reports empty, and a banked save writes into the bank.
    //    The bank survives a factory-then-user load sequence.
    // ----------------------------------------------------------------
    g_section = "bank-identity";
    {
        StubProcessor proc;
        PresetManager pm (proc, proc.apvts);
        pm.getStore().setRootDirectory (tempRoot);

        const auto bankDir = tempRoot.getChildFile ("MyBank");
        bankDir.createDirectory();
        const auto bankFile = bankDir.getChildFile ("Banked.chronos");
        const auto rootFile = tempRoot.getChildFile ("Rooted.chronos");

        {
            MemoryBlock block;
            proc.getStateInformation (block);
            const auto xml = AudioProcessor::getXmlFromBinary (block.getData(), (int) block.getSize());
            CHECK (xml != nullptr);
            CHECK (PresetStore::savePresetFile (bankFile, xml->toString()));
            CHECK (PresetStore::savePresetFile (rootFile, xml->toString()));
        }

        CHECK (store.bankForFile (bankFile) == "MyBank");
        CHECK (store.bankForFile (rootFile).isEmpty());
        CHECK (store.bankForFile (tempRoot.getChildFile ("nope.chronos")).isEmpty());

        // A factory preset, then a root user preset: the bank is empty,
        // not the stale factory bank.
        CHECK (kNumFactoryPresets > 0);
        CHECK (pm.loadFactoryPreset (kFactoryPresets[0].name, kFactoryPresets[0].bank));
        pm.clearModified();
        CHECK (pm.getCurrentBank() == String (kFactoryPresets[0].bank));
        CHECK (pm.loadPreset (rootFile));
        pm.clearModified();
        CHECK (pm.getCurrentBank().isEmpty());

        // A banked load, then a save: the new file lands in the bank.
        CHECK (pm.loadPreset (bankFile));
        pm.clearModified();
        CHECK (pm.getCurrentBank() == "MyBank");
        CHECK (pm.saveAs ("Banked2", "a", "c"));
        CHECK (store.presetFile ("MyBank", "Banked2").existsAsFile());
    }

    // ----------------------------------------------------------------
    // 8. Total recall: a full preset with one PARAM removed is refused
    //    by name, and the state stays untouched.
    // ----------------------------------------------------------------
    g_section = "missing-param";
    {
        StubProcessor proc;
        PresetManager pm (proc, proc.apvts);
        pm.getStore().setRootDirectory (tempRoot);

        MemoryBlock block;
        proc.getStateInformation (block);
        const auto xml = AudioProcessor::getXmlFromBinary (block.getData(), (int) block.getSize());
        CHECK (xml != nullptr);

        bool removed = false;
        for (int i = xml->getNumChildElements() - 1; i >= 0; --i)
        {
            auto* el = xml->getChildElement (i);
            if (el != nullptr && el->hasTagName ("PARAM")
                && el->getStringAttribute ("id") == "mix")
            {
                xml->removeChildElement (el, true);
                removed = true;
                break;
            }
        }
        CHECK (removed);

        const auto file = tempRoot.getChildFile ("MissingMix.chronos");
        CHECK (PresetStore::savePresetFile (file, xml->toString()));

        setDenorm (proc.apvts, "delayTime", 500.0f);
        const float before = getDenorm (proc.apvts, "delayTime");
        const String beforeName = pm.getCurrentName();

        CHECK (! pm.loadPreset (file));
        CHECK (pm.getLastError().contains ("mix"));

        const float after = getDenorm (proc.apvts, "delayTime");
        CHECK (std::fabs (after - before) < 1e-6f);
        CHECK (pm.getCurrentName() == beforeName);
    }

    // ----------------------------------------------------------------
    // 9. Name hygiene: reserved device names and an empty result are
    //    refused, the reserved characters are replaced, and the bank
    //    goes through the same sanitiser.
    // ----------------------------------------------------------------
    g_section = "name-hygiene";
    {
        CHECK (PresetStore::sanitiseName ("CON").isEmpty());
        CHECK (PresetStore::sanitiseName ("con").isEmpty());
        CHECK (PresetStore::sanitiseName ("Com1.chronos").isEmpty());
        CHECK (PresetStore::sanitiseName ("   ").isEmpty());
        CHECK (PresetStore::sanitiseName ("a:b").containsChar (':') == false);
        CHECK (! PresetStore::sanitiseName ("a*b?c\"d<e>f|g")
                    .containsAnyOf ("*?\"<>|:"));

        StubProcessor proc;
        PresetManager pm (proc, proc.apvts);
        pm.getStore().setRootDirectory (tempRoot);
        CHECK (! pm.saveAs ("CON", "a", "c"));

        // A reserved bank name falls back to the root. A bank with a
        // reserved character lands in the sanitised directory.
        CHECK (store.presetFile ("CON", "X") == tempRoot.getChildFile ("X.chronos"));
        CHECK (store.presetFile ("My:Bank", "Y")
               == tempRoot.getChildFile ("My_Bank").getChildFile ("Y.chronos"));
    }

    // ----------------------------------------------------------------
    // 10. A preset carries no geometry. The session side tree holds a
    //     1600 width and a non-default tab; the saved file has neither,
    //     and a load into a side tree at 800 leaves it at 800.
    // ----------------------------------------------------------------
    g_section = "no-geometry";
    {
        StubProcessor proc;
        PresetManager pm (proc, proc.apvts);
        pm.getStore().setRootDirectory (tempRoot);

        proc.editorSide.setProperty ("editorWidth", 1600, nullptr);
        proc.editorSide.setProperty ("timeTab", 1, nullptr);

        CHECK (pm.saveAs ("Geom", "a", "c"));
        const auto file = store.presetFile ({}, "Geom");
        const auto saved = parseXML (file.loadFileAsString());
        CHECK (saved != nullptr);
        CHECK (saved->getChildByName ("EDITOR") == nullptr);
        CHECK (! saved->hasAttribute ("editorWidth"));

        StubProcessor procB;
        PresetManager pmB (procB, procB.apvts);
        pmB.getStore().setRootDirectory (tempRoot);
        procB.editorSide.setProperty ("editorWidth", 800, nullptr);
        procB.editorSide.setProperty ("timeTab", 0, nullptr);

        CHECK (pmB.loadPreset (file));
        pmB.clearModified();
        CHECK (static_cast<int> (procB.editorSide.getProperty ("editorWidth")) == 800);
        CHECK (static_cast<int> (procB.editorSide.getProperty ("timeTab")) == 0);
    }

    // Clean up the temporary directory.
    tempRoot.deleteRecursively();

    std::println("=== preset_store_check OK ===");
    return 0;
}
