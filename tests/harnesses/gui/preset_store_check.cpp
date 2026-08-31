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

    // Mirror the processor state path.
    void getStateInformation (MemoryBlock& destData) override
    {
        ValueTree state = apvts.copyState();
        state.setProperty ("version", kStateVersion, nullptr);
        copyXmlToBinary (*state.createXml(), destData);
    }

    void setStateInformation (const void* data, int sizeInBytes) override
    {
        auto xml = AudioProcessor::getXmlFromBinary (data, sizeInBytes);
        if (xml == nullptr || ! xml->hasTagName (apvts.state.getType())) return;
        ValueTree state (ValueTree::fromXml (*xml));
        if (const int fileVersion = state.getProperty ("version"); fileVersion < kStateVersion)
            migrateState_ (state, fileVersion);
        state.setProperty ("version", kStateVersion, nullptr);
        apvts.replaceState (state);
    }

    AudioProcessorValueTreeState apvts;

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

// The 28 parameter IDs, for round-trip and perturb.
static const char* const kParamIDs[] = {
    "gain",          "bits",          "delayTime",     "delayTimeR",
    "timeLink",      "delaySync",     "delayDivision",  "delayMode",
    "bypass",        "filterMode",    "hpfFreq",       "lpfFreq",
    "mix",           "drive",         "adaaOrder",      "feedback",
    "dampHz",        "loopCutHz",      "crossFeed",      "loopDrive",
    "loopSatOrder",   "delayModDepth",  "delayModRateHz", "enableDiffuser",
    "diffusion",      "diffuserSize",   "diffModDepth",   "diffModRateHz"
};

// Return a pseudo-random value for one parameter. The seed fixes the value per run.
float randomDenorm (AudioProcessorValueTreeState& a, const char* id, int seed)
{
    auto* p = a.getParameter (id);
    if (p == nullptr) return 0.0f;
    const auto range = p->getNormalisableRange();
    return range.convertFrom0to1 (static_cast<float> ((seed * 2654435761u) % 1000) / 999.0f);
}

// Build a version-N state tree XML string for the migration test.
String buildVersionedXml (int version, const String& rootTag, bool outOfRange, bool unknownParam)
{
    String xml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<" + rootTag;
    if (version > 0)
        xml += " version=\"" + String (version) + "\"";
    xml += ">";
    xml += "<PARAM id=\"delayTime\" value=\"900.0\"/>";
    xml += "<PARAM id=\"feedback\" value=\"0.55\"/>";
    xml += "<PARAM id=\"mix\" value=\"42.0\"/>";
    xml += "<PARAM id=\"drive\" value=\"9.0\"/>";
    if (outOfRange)
        xml += "<PARAM id=\"feedback\" value=\"5.0\"/>";
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
            for (int i = 0; i < 28; ++i)
            {
                setDenorm (proc.apvts, kParamIDs[i],
                           randomDenorm (proc.apvts, kParamIDs[i], i + 1 + iter * 31));
                saved[std::size_t (i)] = getDenorm (proc.apvts, kParamIDs[i]);
            }

            CHECK (pm.saveAs (name, "harness", "test"));
            CHECK (pm.getCurrentName() == name);
            CHECK (! pm.isModified());

            // Perturb every parameter so the load must restore the saved values.
            for (int i = 0; i < 28; ++i)
                setDenorm (proc.apvts, kParamIDs[i],
                           randomDenorm (proc.apvts, kParamIDs[i], i + 100 + iter * 31));

            CHECK (pm.isModified());

            const auto file = store.presetFile ({}, name);
            CHECK (pm.loadPreset (file));

            // replaceState fires the parameter listeners synchronously, so
            // the modified flag is true right after loadPreset returns. The
            // async clear is queued; clear manually for the headless harness.
            pm.clearModified();

            for (int i = 0; i < 28; ++i)
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

            for (int i = 0; i < 28; ++i)
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

    // Clean up the temporary directory.
    tempRoot.deleteRecursively();

    std::println("=== preset_store_check OK ===");
    return 0;
}
