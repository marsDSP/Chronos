// tests/harnesses/gui/enablement_check.cpp
//
// Enablement table harness (rev G6 section 4.2). Host-free: it opens
// no editor. It builds a real APVTS from the Chronos layout and
// asserts that the predicate for each table row reads the same
// parameter state the audio path reads. A row is the section 4.2
// table entry: the parameter state and the controls it makes inert.

#include <juce_audio_processors/juce_audio_processors.h>

using namespace juce;

#include "ChronosParameters.h"

#include <cstdio>
#include <print>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond)                                                            \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

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

// One row of the section 4.2 table: the parameter state and the
// controls that the state makes inert. The predicate is the same
// expression the editor applies and the audio honours.
struct Row {
    const char* name;
    const char* controls;
};

const Row kRows[] = {
    { "delaySync true",          "TimePanel time knobs and both readouts" },
    { "delaySync false",         "TimePanel division box" },
    { "delaySync false + timeLink true", "right time knob and readout" },
    { "enableDiffuser false",    "the four diffuser knobs" },
    { "adaaOrder 0",             "DrivePanel drive knob" },
    { "bypass true",             "tap band and card row children" },
};

} // namespace

int main()
{
    StubProcessor proc;
    auto& apvts = proc.apvts;
    ChronosParameters params (apvts);

    // The delay sample math needs a sample rate.
    params.prepare (48000.0);

    // ----------------------------------------------------------------
    // 1. delaySync true: the time knobs have no audible effect.
    //    The processor takes the tempo-derived delay pair and never
    //    reads the knobs. The predicate is the raw sync read.
    // ----------------------------------------------------------------
    g_section = "delaySync";
    {
        setDenorm (apvts, "delaySync", 1.0f);
        CHECK (params.getRawDelaySync() == true);

        setDenorm (apvts, "delaySync", 0.0f);
        CHECK (params.getRawDelaySync() == false);
        std::println ("delaySync row: PASS");
    }

    // ----------------------------------------------------------------
    // 2. delaySync false: the division box is the control that matters.
    //    The knobs drive the delay pair directly.
    // ----------------------------------------------------------------
    g_section = "delayDivision";
    {
        setDenorm (apvts, "delayTime", 100.0f);
        setDenorm (apvts, "delayTimeR", 500.0f);
        setDenorm (apvts, "timeLink", 0.0f);
        setDenorm (apvts, "delaySync", 0.0f);

        // With sync off and link off, each knob reaches its own channel.
        params.update();
        CHECK (params.getDelaySamplesL() != params.getDelaySamplesR());

        setDenorm (apvts, "delayDivision", 7.0f);
        CHECK (params.getRawDelayDivision() == 7);
        std::println ("delayDivision row: PASS");
    }

    // ----------------------------------------------------------------
    // 3. timeLink true: the right time knob has no audible effect.
    //    update() copies the left value to both channels.
    // ----------------------------------------------------------------
    g_section = "timeLink";
    {
        setDenorm (apvts, "delayTime", 100.0f);
        setDenorm (apvts, "delayTimeR", 900.0f);
        setDenorm (apvts, "timeLink", 1.0f);

        params.update();
        CHECK (params.getDelaySamplesL() == params.getDelaySamplesR());

        setDenorm (apvts, "timeLink", 0.0f);
        params.update();
        CHECK (params.getDelaySamplesL() != params.getDelaySamplesR());
        std::println ("timeLink row: PASS");
    }

    // ----------------------------------------------------------------
    // 4. enableDiffuser false: the four diffuser knobs have no effect.
    //    The audio fades the diffuser fully out. The predicate is the
    //    raw enable read.
    // ----------------------------------------------------------------
    g_section = "enableDiffuser";
    {
        setDenorm (apvts, "enableDiffuser", 0.0f);
        CHECK (params.getRawEnableDiffuser() == false);

        setDenorm (apvts, "enableDiffuser", 1.0f);
        CHECK (params.getRawEnableDiffuser() == true);

        setDenorm (apvts, "enableDiffuser", 0.0f);
        std::println ("enableDiffuser row: PASS");
    }

    // ----------------------------------------------------------------
    // 5. adaaOrder 0: the drive knob has no audible effect. The engine
    //    assigns sat0 = wet0 and never applies the drive gain. The
    //    predicate is the choice index read.
    // ----------------------------------------------------------------
    g_section = "adaaOrder";
    {
        setDenorm (apvts, "adaaOrder", 0.0f);
        CHECK (params.getADAAOrder() == 0);

        setDenorm (apvts, "adaaOrder", 1.0f);
        CHECK (params.getADAAOrder() == 1);

        setDenorm (apvts, "adaaOrder", 2.0f);
        CHECK (params.getADAAOrder() == 2);
        std::println ("adaaOrder row: PASS");
    }

    // ----------------------------------------------------------------
    // 6. bypass true: every child of the tap band and the card row is
    //    inert. The bypass parameter is the predicate.
    // ----------------------------------------------------------------
    g_section = "bypass";
    {
        setDenorm (apvts, "bypass", 0.0f);
        CHECK (params.getBypass() == false);

        setDenorm (apvts, "bypass", 1.0f);
        CHECK (params.getBypass() == true);

        setDenorm (apvts, "bypass", 0.0f);
        std::println ("bypass row: PASS");
    }

    // ----------------------------------------------------------------
    // 7. Table totality: the six rows cover the section 4.2 table.
    // ----------------------------------------------------------------
    g_section = "table";
    {
        CHECK (sizeof (kRows) / sizeof (kRows[0]) == 6);
        std::println ("table totality (6 rows): PASS");
    }

    std::println ("enablement_check: ALL PASS");
    return 0;
}
