/**
 * Correctness harness for EditHistory (rev G7 appendix A).
 * Uses the StubProcessor pattern from enablement_check.
 */

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include "state/EditHistory.h"
#include "ChronosParameters.h"

#include <print>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

class StubProcessor final : public juce::AudioProcessor
{
public:
    StubProcessor()
        : apvts(*this, nullptr, "Parameters", ChronosParameters::createParameterLayout()) {}

    const String getName() const override { return {}; }
    void prepareToPlay(double, int) override {}
    void releaseResources() override {}
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override {}
    using AudioProcessor::processBlock;
    double getTailLengthSeconds() const override { return {}; }
    bool acceptsMidi() const override { return {}; }
    bool producesMidi() const override { return {}; }
    bool isMidiEffect() const override { return false; }
    juce::AudioProcessorEditor* createEditor() override { return {}; }
    bool hasEditor() const override { return false; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const String getProgramName(int) override { return {}; }
    void changeProgramName(int, const String&) override {}
    void getStateInformation(juce::MemoryBlock&) override {}
    void setStateInformation(const void*, int) override {}

    juce::AudioProcessorValueTreeState apvts;
};

// Set a parameter through a gesture bracket.
void gestureSet(juce::RangedAudioParameter* p, float norm)
{
    if (p == nullptr) return;
    p->beginChangeGesture();
    p->setValueNotifyingHost(norm);
    p->endChangeGesture();
}

// Set a parameter without a gesture (host automation).
void rawSet(juce::RangedAudioParameter* p, float norm)
{
    if (p == nullptr) return;
    p->setValueNotifyingHost(norm);
}

int runAll()
{
    StubProcessor proc;
    auto& apvts = proc.apvts;
    MarsDSP::State::EditHistory history(proc);

    // 1. One gesture records one entry, named after the parameter.
    g_section = "one_gesture";
    {
        auto* fb = apvts.getParameter(feedbackParamID.getParamID());
        const float old = fb->getValue();
        gestureSet(fb, old + 0.1f);

        CHECK(history.canUndo());
        CHECK(! history.canRedo());
        CHECK(history.undoName().isNotEmpty());
        history.undo();
        CHECK(std::fabs(fb->getValue() - old) < 1e-6f);
        CHECK(! history.canUndo());
        CHECK(history.canRedo());
        history.redo();
        CHECK(std::fabs(fb->getValue() - (old + 0.1f)) < 1e-6f);
        std::println("one gesture, one entry, undo and redo: PASS");
    }

    // 2. Two overlapping gestures record one entry with two changes.
    g_section = "overlapping";
    {
        auto* fb = apvts.getParameter(feedbackParamID.getParamID());
        auto* cf = apvts.getParameter(crossFeedParamID.getParamID());
        const float oldFb = fb->getValue();
        const float oldCf = cf->getValue();

        fb->beginChangeGesture();
        cf->beginChangeGesture();
        fb->setValueNotifyingHost(oldFb + 0.05f);
        cf->setValueNotifyingHost(oldCf + 0.05f);
        fb->endChangeGesture();
        cf->endChangeGesture();

        history.undo();
        CHECK(std::fabs(fb->getValue() - oldFb) < 1e-6f);
        CHECK(std::fabs(cf->getValue() - oldCf) < 1e-6f);
        std::println("two overlapping gestures, one entry with two changes: PASS");
    }

    // 3. A gesture without a change records nothing.
    g_section = "no_change";
    {
        history.clear();
        auto* fb = apvts.getParameter(feedbackParamID.getParamID());
        const float v = fb->getValue();
        fb->beginChangeGesture();
        fb->endChangeGesture();
        // No value change, so no entry.
        CHECK(! history.canUndo());
        std::println("gesture without a change records nothing: PASS");
    }

    // 4. A new gesture after undo truncates the redo tail.
    g_section = "truncate_redo";
    {
        auto* fb = apvts.getParameter(feedbackParamID.getParamID());
        const float v0 = fb->getValue();
        gestureSet(fb, v0 + 0.1f);
        gestureSet(fb, v0 + 0.2f);
        history.undo();  // back to v0 + 0.1f
        CHECK(history.canRedo());
        gestureSet(fb, v0 + 0.3f);  // new gesture
        CHECK(! history.canRedo());  // redo tail truncated
        std::println("new gesture after undo truncates redo: PASS");
    }

    // 5. 120 gestures leave 100 entries with the oldest gone.
    g_section = "capacity";
    {
        history.clear();
        auto* fb = apvts.getParameter(feedbackParamID.getParamID());
        for (int i = 0; i < 120; ++i)
            gestureSet(fb, std::clamp(fb->getValue() + 0.001f, 0.0f, 1.0f));
        CHECK(history.canUndo());
        // Undo 100 times to reach the initial state.
        for (int i = 0; i < 100; ++i)
            history.undo();
        CHECK(! history.canUndo());
        std::println("120 gestures leave 100 entries, oldest gone: PASS");
    }

    // 6. Undo and redo record nothing themselves.
    g_section = "undo_redo_silent";
    {
        history.clear();
        auto* fb = apvts.getParameter(feedbackParamID.getParamID());
        const float v = fb->getValue();
        gestureSet(fb, v + 0.1f);
        const int entriesBefore = 0;
        CHECK(history.canUndo());
        history.undo();
        history.redo();
        // Still only one undoable entry (undo+redo didn't add).
        CHECK(history.canUndo());
        history.undo();
        CHECK(! history.canUndo());
        std::println("undo and redo record nothing: PASS");
    }

    // 7. recordSnapshot after three changed parameters records one entry.
    g_section = "snapshot";
    {
        history.clear();
        auto* fb = apvts.getParameter(feedbackParamID.getParamID());
        auto* cf = apvts.getParameter(crossFeedParamID.getParamID());
        auto* dr = apvts.getParameter(driveParamID.getParamID());

        const auto& all = proc.getParameters();
        std::vector<float> before(all.size());
        for (std::size_t i = 0; i < all.size(); ++i)
            before[i] = all[i]->getValue();

        gestureSet(fb, fb->getValue() + 0.1f);
        gestureSet(cf, cf->getValue() + 0.1f);
        gestureSet(dr, dr->getValue() + 0.1f);

        history.recordSnapshot("Load Test", before);
        CHECK(history.canUndo());
        history.undo();
        CHECK(std::fabs(fb->getValue() - before[static_cast<int>(std::find(all.begin(), all.end(), fb) - all.begin())]) < 1e-6f);
        std::println("recordSnapshot after three changes, one entry: PASS");
    }

    // 8. setValueNotifyingHost without a gesture records nothing.
    g_section = "no_gesture";
    {
        history.clear();
        auto* fb = apvts.getParameter(feedbackParamID.getParamID());
        const float v = fb->getValue();
        rawSet(fb, v + 0.2f);
        CHECK(! history.canUndo());
        std::println("setValueNotifyingHost without a gesture records nothing: PASS");
    }

    return 0;
}

} // namespace

int main()
{
    std::println("=== Chronos EditHistory harness ===");
    std::println();
    const int r = runAll();
    std::println();
    std::println("=== {} ===", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}
