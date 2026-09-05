#include "ChronosEditor.h"
#include "gui/controls/SegmentButtons.h"
#include "gui/MetricsConsumer.h"

using Metrics = MarsDSP::GUI::Metrics;

namespace {

using namespace MarsDSP::GUI::Knobs;
using GUIColours = MarsDSP::GUI::Colours;
using MarsDSP::GUI::Metrics;
using MarsDSP::GUI::AccentConsumer;
using MarsDSP::GUI::MetricsConsumer;
using MarsDSP::GUI::EnablementConsumer;
using MarsDSP::GUI::EnablementState;
// Tooltip delay in milliseconds (section 4.5).
constexpr int kTooltipDelayMs = 700;

// A timer that calls a function on the message thread at a fixed rate.
class LambdaTimer : public juce::Timer {
public:
    explicit LambdaTimer(std::function<void()> cb, int hz)
        : cb_(std::move(cb)) { startTimerHz(hz); }
    ~LambdaTimer() override { stopTimer(); }
    void timerCallback() override { cb_(); }
private:
    std::function<void()> cb_;
};

// Derive the knob diameter in pixels for n knobs in a content area.
// rowH is the pixel height of the knob row (label band plus gap plus knob).
// hasReadout accounts for a value readout below the knob row.
float knobDiameterPx(const Metrics& m, const float contentW, const float rowH, const int n,
                    const bool hasReadout)
{
    const float g = m.pxf(static_cast<float>(Metrics::kKnobGutter));
    const float cellW = (contentW - static_cast<float>(n - 1) * g) / static_cast<float>(n);
    const float cellH = rowH
        - static_cast<float>(m.px(static_cast<float>(Metrics::kLabelBandH)))
        - static_cast<float>(m.px(static_cast<float>(Metrics::kKnobLabelGap)))
        - (hasReadout
               ? static_cast<float>(m.px(static_cast<float>(Metrics::kReadoutBandH)))
                 + static_cast<float>(m.px(static_cast<float>(Metrics::kLabelReadoutGap)))
               : 0.0f);
    return std::clamp(std::min(cellW, cellH),
                      m.pxf(static_cast<float>(Metrics::kKnobMin)),
                      m.pxf(static_cast<float>(Metrics::kKnobMax)));
}

// Return the PDLKnob cell height in pixels: label band + gap + knob.
int knobCellHeightPx(const Metrics& m, const int dPx)
{
    return m.px(static_cast<float>(Metrics::kLabelBandH))
         + m.px(static_cast<float>(Metrics::kKnobLabelGap))
         + dPx;
}

// Return the core accent for the current delay mode.
Colour coreAccent(const ChronosProcessor& proc)
{
    return (proc.getParameters().getRawDelayMode() == 1) ? GUIColours::accentDelayBBD
                                                          : GUIColours::accentDelayDigital;
}

// 1. TIME card. Absorbs the MOD panel.
class TimePanel final : public Component, public AccentConsumer, public MetricsConsumer,
                        public EnablementConsumer {
public:
    explicit TimePanel(ChronosProcessor& proc, PedalKnob& knobLnf)
        : timeLKnob("LEFT TIME", proc.getAPVTS(), delayTimeParamID, knobLnf),
          timeRKnob("RIGHT TIME", proc.getAPVTS(), delayTimeRParamID, knobLnf),
          depthKnob("MOD DEPTH", proc.getAPVTS(), delayModDepthParamID, knobLnf),
          rateKnob("MOD RATE", proc.getAPVTS(), delayModRateHzParamID, knobLnf)
    {
    timeLDisplay.setSlider(&timeLKnob.getSlider(),
                           proc.getAPVTS().getParameter(delayTimeParamID.getParamID()));
    timeRDisplay.setSlider(&timeRKnob.getSlider(),
                           proc.getAPVTS().getParameter(delayTimeRParamID.getParamID()));
    addAndMakeVisible(timeLDisplay);
    addAndMakeVisible(timeRDisplay);
    addAndMakeVisible(timeLKnob);
    addAndMakeVisible(timeRKnob);

    timeLKnob.setTooltip("Set the left channel delay time. Range 1 to 5000 milliseconds.");
    timeRKnob.setTooltip("Set the right channel delay time. Range 1 to 5000 milliseconds.");
    timeLDisplay.setTooltip("Drag to adjust the left delay time.");
    timeRDisplay.setTooltip("Drag to adjust the right delay time.");
    timeLDisplay.setTitle("Left Delay Time");
    timeRDisplay.setTitle("Right Delay Time");
    timeLDisplay.setHelpText("Drag to adjust the left delay time.");
    timeRDisplay.setHelpText("Drag to adjust the right delay time.");

    timeLinkButton.setColours(coreAccent(proc), GUIColours::textMuted);
    timeLinkButton.setTooltip("Link the left and right delay times.");
    timeLinkButton.setTitle("Time Link");
    timeLinkButton.setHelpText("Link the left and right delay times.");
        timeLinkAttach = std::make_unique<AudioProcessorValueTreeState::ButtonAttachment>(
            proc.getAPVTS(), timeLinkParamID.getParamID(), timeLinkButton);
        addAndMakeVisible(timeLinkButton);

    syncButton.setMusicalNote(true);
    syncButton.setColours(coreAccent(proc), GUIColours::textMuted);
    syncButton.setTooltip("Sync the delay time to the host tempo.");
    syncButton.setTitle("Tempo Sync");
    syncButton.setHelpText("Sync the delay time to the host tempo.");
        syncAttach = std::make_unique<AudioProcessorValueTreeState::ButtonAttachment>(
            proc.getAPVTS(), delaySyncParamID.getParamID(), syncButton);
        addAndMakeVisible(syncButton);

        static const StringArray divisions = {
            "1/64", "1/32T", "1/32", "1/16T", "1/32.", "1/16",
            "1/8T", "1/16.", "1/8", "1/4T", "1/8.", "1/4",
            "1/2T", "1/4.", "1/2", "1/1T", "1/2.", "1/1",
            "2/1", "4/1"
        };
        divisionBox.addItemList(divisions, 1);
    divisionBox.setTooltip("Select the tempo-sync division.");
    divisionBox.setTitle("Delay Division");
    divisionBox.setHelpText("Select the tempo-sync division.");

        divisionBox.onChange = [this]
        {
            const int div = divisionBox.getSelectedId() - 1;
            timeLDisplay.setDivision(div);
            timeRDisplay.setDivision(div);
        };

        divisionAttach = std::make_unique<AudioProcessorValueTreeState::ComboBoxAttachment>(
            proc.getAPVTS(), delayDivisionParamID.getParamID(), divisionBox);
        addAndMakeVisible(divisionBox);

    addAndMakeVisible(depthKnob);
    addAndMakeVisible(rateKnob);
    depthKnob.setTooltip("Set the delay pitch modulation depth. Range 0 to 50 cents.");
    rateKnob.setTooltip("Set the delay modulation rate. Range 0.01 to 10 hertz.");
    }

    void resized() override
    {
        const auto m = metrics_;
        const float w = static_cast<float>(getWidth());

        const float g = m.pxf(static_cast<float>(Metrics::kKnobGutter));
        const int gapPx = roundToInt(g);
        const int knobRowH = m.px(static_cast<float>(Metrics::kKnobRowH));
        const int labelReadoutGap = m.px(static_cast<float>(Metrics::kLabelReadoutGap));
        const int readoutH = m.px(static_cast<float>(Metrics::kReadoutBandH));
        const int interRowGap = m.px(static_cast<float>(Metrics::kInterRowGap));
        const int selH = m.px(Metrics::kSelectorRowH);

        const int cellWPx = roundToInt((w - g) / 2.0f);

        // Row 1: two time knobs.
        const float d = knobDiameterPx(m, w, static_cast<float>(knobRowH), 2, false);
        const int cellH = knobCellHeightPx(m, roundToInt(d));
        timeLKnob.setBounds(0, 0, cellWPx, cellH);
        timeRKnob.setBounds(cellWPx + gapPx, 0, cellWPx, cellH);

        // Readouts below the knob row.
        int y = knobRowH + labelReadoutGap;
        timeLDisplay.setBounds(0, y, cellWPx, readoutH);
        timeRDisplay.setBounds(cellWPx + gapPx, y, cellWPx, readoutH);

        // Row 2: the selector row.
        y += readoutH + interRowGap;
        const int btnSize = m.px(Metrics::kToggleSize);
        const int btnGap = m.px(Metrics::kToggleGap);
        int sx = 0;
        timeLinkButton.setBounds(sx, y, btnSize, btnSize);  sx += btnSize + btnGap;
        syncButton.setBounds(sx, y, btnSize, btnSize);       sx += btnSize + btnGap;
        divisionBox.setBounds(sx, y + m.px(Metrics::kSelectorNudge), getWidth() - sx, selH);

        // Row 3: two mod knobs.
        y += selH + interRowGap;
        const float d2 = knobDiameterPx(m, w, static_cast<float>(knobRowH), 2, false);
        const int cellH2 = knobCellHeightPx(m, roundToInt(d2));
        depthKnob.setBounds(0, y, cellWPx, cellH2);
        rateKnob.setBounds(cellWPx + gapPx, y, cellWPx, cellH2);
    }

    void setAccentColour(Colour c) override
    {
        timeLDisplay.setAccentColour(c);
        timeRDisplay.setAccentColour(c);
        timeLinkButton.setAccentColour(c);
        syncButton.setAccentColour(c);
        timeLKnob.setAccentColour(c);
        timeRKnob.setAccentColour(c);
        depthKnob.setAccentColour(c);
        rateKnob.setAccentColour(c);
    }

    void setMetrics(const Metrics& m) override
    {
        metrics_ = m;
        timeLKnob.setMetrics(m);
        timeRKnob.setMetrics(m);
        timeLDisplay.setMetrics(m);
        timeRDisplay.setMetrics(m);
        timeLinkButton.setMetrics(m);
        syncButton.setMetrics(m);
        depthKnob.setMetrics(m);
        rateKnob.setMetrics(m);
        resized();
    }

    void setControlsEnabled(const EnablementState& state) override
    {
        const bool leftLive = ! state.delaySync;
        const bool rightLive = ! state.delaySync && ! state.timeLink;
        const bool divisionLive = state.delaySync;

        timeLKnob.setEnabled(leftLive);
        timeLDisplay.setEnabled(leftLive);
        timeRKnob.setEnabled(rightLive);
        timeRDisplay.setEnabled(rightLive);

        divisionBox.setEnabled(divisionLive);
        divisionBox.setAlpha(divisionLive ? 1.0f : MarsDSP::GUI::kInertAlpha);

        const int div = divisionBox.getSelectedId() - 1;
        timeLDisplay.setDivision(div);
        timeRDisplay.setDivision(div);
        timeLDisplay.setSyncState(state.delaySync);
        timeRDisplay.setSyncState(state.delaySync);
    }

private:
    Metrics metrics_;
    PDLKnob timeLKnob;
    PDLKnob timeRKnob;
    PDLKnob depthKnob;
    PDLKnob rateKnob;
    MarsDSP::GUI::TimeDisplay timeLDisplay;
    MarsDSP::GUI::TimeDisplay timeRDisplay;
    MarsDSP::GUI::TimeLockButton timeLinkButton;
    MarsDSP::GUI::PowerButton syncButton;
    ComboBox divisionBox;
    std::unique_ptr<AudioProcessorValueTreeState::ButtonAttachment> timeLinkAttach;
    std::unique_ptr<AudioProcessorValueTreeState::ButtonAttachment> syncAttach;
    std::unique_ptr<AudioProcessorValueTreeState::ComboBoxAttachment> divisionAttach;
};

// 2. REPEATS card. Absorbs the TONE panel.
class RepeatsPanel final : public Component, public AccentConsumer, public MetricsConsumer {
public:
    explicit RepeatsPanel(ChronosProcessor& proc, PedalKnob& knobLnf)
        : feedbackKnob("FEEDBACK", proc.getAPVTS(), feedbackParamID, knobLnf),
          crossFeedKnob("CROSS", proc.getAPVTS(), crossFeedParamID, knobLnf),
          loopDriveKnob("LOOP DRIVE", proc.getAPVTS(), loopDriveParamID, knobLnf),
          dampKnob("DAMP", proc.getAPVTS(), dampHzParamID, knobLnf),
          loopCutKnob("LOW CUT", proc.getAPVTS(), loopCutHzParamID, knobLnf),
          loopSatSeg_(proc.getAPVTS(), loopSatOrderParamID.getParamID(),
                      StringArray{"Off", "1st", "2nd"}, coreAccent(proc), true,
                      MarsDSP::GUI::kAntiAliasLabels, MarsDSP::GUI::kAntiAliasTooltips),
          delayModeSeg_(proc.getAPVTS(), delayModeParamID.getParamID(),
                        StringArray{"Digital", "BBD"}, coreAccent(proc), true)
    {
        addAndMakeVisible(feedbackKnob);
        addAndMakeVisible(crossFeedKnob);
        addAndMakeVisible(loopDriveKnob);
        addAndMakeVisible(dampKnob);
        addAndMakeVisible(loopCutKnob);
        addAndMakeVisible(loopSatSeg_);
        addAndMakeVisible(delayModeSeg_);
        feedbackKnob.setTooltip("Set the level of the repeats fed back into the loop. Range 0 to 115 percent.");
        crossFeedKnob.setTooltip("Set the cross-channel feedback level. Range 0 to 100 percent.");
        loopDriveKnob.setTooltip("Set the repeat loop input drive. Range minus 6 to 24 decibels.");
        dampKnob.setTooltip("Set the repeat damping cutoff. Range 200 to 20000 hertz.");
        loopCutKnob.setTooltip("Set the repeat loop low cut. Range 20 to 2000 hertz.");
        delayModeSeg_.setTooltip("Select the delay core type.");

        // The loop drive arc grows from the 0 dB angle in either direction.
        if (auto* p = proc.getAPVTS().getParameter(loopDriveParamID.getParamID()))
            loopDriveKnob.setArcOrigin(static_cast<float>(p->getNormalisableRange().convertTo0to1(0.0f)));
    }

    void resized() override
    {
        const auto m = metrics_;
        const float w = static_cast<float>(getWidth());

        const float g = m.pxf(static_cast<float>(Metrics::kKnobGutter));
        const int gapPx = roundToInt(g);
        const int knobRowH = m.px(static_cast<float>(Metrics::kKnobRowH));
        const int interRowGap = m.px(static_cast<float>(Metrics::kInterRowGap));
        const int selH = m.px(Metrics::kSelectorRowH);

        // Row 1: the delay-mode segment.
        int y = 0;
        delayModeSeg_.setBounds(0, y, getWidth(), selH);

        // Row 2: three knobs.
        y += selH + interRowGap;
        const int cellW3 = roundToInt((w - 2.0f * g) / 3.0f);
        const float d3 = knobDiameterPx(m, w, static_cast<float>(knobRowH), 3, false);
        const int cellH3 = knobCellHeightPx(m, roundToInt(d3));
        int x = 0;
        feedbackKnob.setBounds(x, y, cellW3, cellH3);   x += cellW3 + gapPx;
        crossFeedKnob.setBounds(x, y, cellW3, cellH3);   x += cellW3 + gapPx;
        loopDriveKnob.setBounds(x, y, cellW3, cellH3);

        // Row 3: two knobs, centred.
        y += knobRowH + interRowGap;
        const int cellW2 = roundToInt((w - g) / 2.0f);
        const float d2 = knobDiameterPx(m, w, static_cast<float>(knobRowH), 2, false);
        const int cellH2 = knobCellHeightPx(m, roundToInt(d2));
        x = 0;
        dampKnob.setBounds(x, y, cellW2, cellH2);   x += cellW2 + gapPx;
        loopCutKnob.setBounds(x, y, cellW2, cellH2);

        // Row 4: the loop-sat segment.
        y += knobRowH + interRowGap;
        loopSatSeg_.setBounds(0, y, getWidth(), selH);
    }

    void setAccentColour(Colour c) override
    {
        loopSatSeg_.setAccentColour(c);
        delayModeSeg_.setAccentColour(c);
        feedbackKnob.setAccentColour(c);
        crossFeedKnob.setAccentColour(c);
        loopDriveKnob.setAccentColour(c);
        dampKnob.setAccentColour(c);
        loopCutKnob.setAccentColour(c);
    }

    void setMetrics(const Metrics& m) override
    {
        metrics_ = m;
        feedbackKnob.setMetrics(m);
        crossFeedKnob.setMetrics(m);
        loopDriveKnob.setMetrics(m);
        dampKnob.setMetrics(m);
        loopCutKnob.setMetrics(m);
        loopSatSeg_.setMetrics(m);
        delayModeSeg_.setMetrics(m);
        resized();
    }

private:
    Metrics metrics_;
    PDLKnob feedbackKnob;
    PDLKnob crossFeedKnob;
    PDLKnob loopDriveKnob;
    PDLKnob dampKnob;
    PDLKnob loopCutKnob;
    MarsDSP::GUI::SegmentButtons loopSatSeg_;
    MarsDSP::GUI::SegmentButtons delayModeSeg_;
};

// 3. DIFFUSER card.
class DiffuserPanel final : public Component, public AccentConsumer, public MetricsConsumer,
                            public EnablementConsumer {
public:
    explicit DiffuserPanel(ChronosProcessor& proc, PedalKnob& knobLnf)
        : modDepthKnob("DIFF MOD", proc.getAPVTS(), diffModDepthParamID, knobLnf),
          modRateKnob("DIFF RATE", proc.getAPVTS(), diffModRateHzParamID, knobLnf),
          pad_(proc.getAPVTS(),
               diffusionParamID.getParamID(),
               diffuserSizeParamID.getParamID(),
               enableDiffuserParamID.getParamID())
    {
        enableButton.setColours(coreAccent(proc), GUIColours::textMuted);
        enableButton.setTooltip("Enable the diffuser section.");
        enableButton.setTitle("Diffuser Enable");
        enableButton.setHelpText("Enable the diffuser section.");
        enableAttach = std::make_unique<AudioProcessorValueTreeState::ButtonAttachment>(
            proc.getAPVTS(), enableDiffuserParamID.getParamID(), enableButton);
        addAndMakeVisible(enableButton);

        addAndMakeVisible(pad_);
        addAndMakeVisible(modDepthKnob);
        addAndMakeVisible(modRateKnob);
        modDepthKnob.setTooltip("Set the diffuser modulation depth. Range 0 to 1.5 milliseconds.");
        modRateKnob.setTooltip("Set the diffuser modulation rate. Range 0.01 to 8 hertz.");
    }

    void resized() override
    {
        const auto m = metrics_;
        const float w = static_cast<float>(getWidth());

        const float g = m.pxf(static_cast<float>(Metrics::kKnobGutter));
        const int gapPx = roundToInt(g);
        const int enableH = m.px(Metrics::kEnableRowH);
        const int enableGap = m.px(Metrics::kEnableGap);
        const int padH = m.px(static_cast<float>(Metrics::kPadH));
        const int knobRowH = m.px(static_cast<float>(Metrics::kKnobRowH));
        const int interRowGap = m.px(static_cast<float>(Metrics::kInterRowGap));
        const int btnSize = m.px(Metrics::kToggleSize);

        // Row 1: the enable button, centred.
        int y = 0;
        enableButton.setBounds((getWidth() - btnSize) / 2, y, btnSize, btnSize);

        // Row 2: the pad.
        y += enableH + enableGap;
        pad_.setBounds(0, y, getWidth(), padH);

        // Row 3: two knobs.
        y += padH + interRowGap;
        const int cellW2 = roundToInt((w - g) / 2.0f);
        const float d2 = knobDiameterPx(m, w, static_cast<float>(knobRowH), 2, false);
        const int cellH2 = knobCellHeightPx(m, roundToInt(d2));
        int x = 0;
        modDepthKnob.setBounds(x, y, cellW2, cellH2);  x += cellW2 + gapPx;
        modRateKnob.setBounds(x, y, cellW2, cellH2);
    }

    void setAccentColour(Colour c) override
    {
        enableButton.setAccentColour(c);
        pad_.setAccentColour(c);
        modDepthKnob.setAccentColour(c);
        modRateKnob.setAccentColour(c);
    }

    void setMetrics(const Metrics& m) override
    {
        metrics_ = m;
        enableButton.setMetrics(m);
        pad_.setMetrics(m);
        modDepthKnob.setMetrics(m);
        modRateKnob.setMetrics(m);
        resized();
    }

    void setControlsEnabled(const EnablementState& state) override
    {
        const bool live = state.enableDiffuser;
        pad_.setEnabled(live);
        modDepthKnob.setEnabled(live);
        modRateKnob.setEnabled(live);
    }

private:
    Metrics metrics_;
    MarsDSP::GUI::PowerButton enableButton;
    MarsDSP::GUI::DiffuserPad pad_;
    PDLKnob modDepthKnob;
    PDLKnob modRateKnob;
    std::unique_ptr<AudioProcessorValueTreeState::ButtonAttachment> enableAttach;
};

// 4. DRIVE card.
class DrivePanel final : public Component, public AccentConsumer, public MetricsConsumer,
                         public EnablementConsumer {
public:
    explicit DrivePanel(ChronosProcessor& proc, PedalKnob& knobLnf)
        : driveKnob("DRIVE", proc.getAPVTS(), driveParamID, knobLnf),
          adaaSeg_(proc.getAPVTS(), adaaOrderParamID.getParamID(),
                   StringArray{"Off", "1st", "2nd"}, coreAccent(proc), false,
                   MarsDSP::GUI::kAntiAliasLabels, MarsDSP::GUI::kAntiAliasTooltips)
    {
        addAndMakeVisible(driveKnob);
        addAndMakeVisible(adaaSeg_);
        driveKnob.setTooltip("Set the output saturator drive. Range 0 to 24 decibels.");
    }

    void resized() override
    {
        const auto m = metrics_;
        const float w = static_cast<float>(getWidth());

        const float g = m.pxf(static_cast<float>(Metrics::kKnobGutter));
        const int gapPx = roundToInt(g);
        const int knobRowH = m.px(static_cast<float>(Metrics::kKnobRowH));
        const int selH = m.px(Metrics::kSelectorRowH);

        // One row: the drive knob at the left, the segment filling the rest.
        const int cellWPx = roundToInt((w - g) / 2.0f);
        const float d = knobDiameterPx(m, w, static_cast<float>(knobRowH), 2, false);
        const int cellH = knobCellHeightPx(m, roundToInt(d));

        driveKnob.setBounds(0, 0, cellWPx, cellH);

        // The segment is selector-row tall, centred on the knob cell.
        const int segY = (cellH - selH) / 2;
        adaaSeg_.setBounds(cellWPx + gapPx, segY, cellWPx, selH);
    }

    void setAccentColour(Colour c) override
    {
        adaaSeg_.setAccentColour(c);
        driveKnob.setAccentColour(c);
    }

    void setMetrics(const Metrics& m) override
    {
        metrics_ = m;
        driveKnob.setMetrics(m);
        adaaSeg_.setMetrics(m);
        resized();
    }

    void setControlsEnabled(const EnablementState& state) override
    {
        driveKnob.setEnabled(! state.driveSatOff);
    }

private:
    Metrics metrics_;
    PDLKnob driveKnob;
    MarsDSP::GUI::SegmentButtons adaaSeg_;
};

// 5. FILTER card.
class FilterPanel final : public Component, public AccentConsumer, public MetricsConsumer {
public:
    explicit FilterPanel(ChronosProcessor& proc, PedalKnob& knobLnf)
        : hpfKnob("HPF", proc.getAPVTS(), hpfFreqParamID, knobLnf),
          lpfKnob("LPF", proc.getAPVTS(), lpfFreqParamID, knobLnf),
          modeSeg_(proc.getAPVTS(), filterModeParamID.getParamID(),
                   StringArray{"Digital", "Analog"}, coreAccent(proc), false)
    {
        addAndMakeVisible(modeSeg_);
        addAndMakeVisible(hpfKnob);
        addAndMakeVisible(lpfKnob);
        hpfKnob.setTooltip("Set the output high-pass cutoff. Range 20 to 2000 hertz.");
        lpfKnob.setTooltip("Set the output low-pass cutoff. Range 200 to 20000 hertz.");
        modeSeg_.setTooltip("Select the output filter type.");
    }

    void resized() override
    {
        const auto m = metrics_;
        const float w = static_cast<float>(getWidth());

        const float g = m.pxf(static_cast<float>(Metrics::kKnobGutter));
        const int gapPx = roundToInt(g);
        const int knobRowH = m.px(static_cast<float>(Metrics::kKnobRowH));
        const int interRowGap = m.px(static_cast<float>(Metrics::kInterRowGap));
        const int selH = m.px(Metrics::kSelectorRowH);

        // Row 1: the filter-mode segment.
        int y = 0;
        modeSeg_.setBounds(0, y, getWidth(), selH);

        // Row 2: two knobs.
        y += selH + interRowGap;
        const int cellWPx = roundToInt((w - g) / 2.0f);
        const float d = knobDiameterPx(m, w, static_cast<float>(knobRowH), 2, false);
        const int cellH = knobCellHeightPx(m, roundToInt(d));
        int x = 0;
        hpfKnob.setBounds(x, y, cellWPx, cellH);  x += cellWPx + gapPx;
        lpfKnob.setBounds(x, y, cellWPx, cellH);
    }

    void setAccentColour(Colour c) override
    {
        modeSeg_.setAccentColour(c);
        hpfKnob.setAccentColour(c);
        lpfKnob.setAccentColour(c);
    }

    void setMetrics(const Metrics& m) override
    {
        metrics_ = m;
        hpfKnob.setMetrics(m);
        lpfKnob.setMetrics(m);
        modeSeg_.setMetrics(m);
        resized();
    }

private:
    Metrics metrics_;
    PDLKnob hpfKnob;
    PDLKnob lpfKnob;
    MarsDSP::GUI::SegmentButtons modeSeg_;
};

// 6. LEVEL card.
class LevelPanel final : public Component, public AccentConsumer, public MetricsConsumer {
public:
    explicit LevelPanel(ChronosProcessor& proc, PedalKnob& knobLnf)
        : mixKnob("MIX", proc.getAPVTS(), mixParamID, knobLnf),
          gainKnob("GAIN", proc.getAPVTS(), gainParamID, knobLnf),
          bitsKnob("BIT DEPTH", proc.getAPVTS(), bitsParamID, knobLnf)
    {
        addAndMakeVisible(mixKnob);
        addAndMakeVisible(gainKnob);
        addAndMakeVisible(bitsKnob);
        mixKnob.setTooltip("Set the dry to wet blend. Range 0 to 100 percent.");
        gainKnob.setTooltip("Set the output gain. Range minus 24 to 12 decibels.");
        bitsKnob.setTooltip("Set the output bit depth. Range 4 to 32 bits.");

        // The gain arc grows from the 0 dB angle in either direction.
        if (auto* p = proc.getAPVTS().getParameter(gainParamID.getParamID()))
            gainKnob.setArcOrigin(static_cast<float>(p->getNormalisableRange().convertTo0to1(0.0f)));
    }

    void resized() override
    {
        const auto m = metrics_;
        const float w = static_cast<float>(getWidth());

        const float g = m.pxf(static_cast<float>(Metrics::kKnobGutter));
        const int gapPx = roundToInt(g);
        const int knobRowH = m.px(static_cast<float>(Metrics::kKnobRowH));
        const int cellWPx = roundToInt((w - 2.0f * g) / 3.0f);

        const float d = knobDiameterPx(m, w, static_cast<float>(knobRowH), 3, false);
        const int cellH = knobCellHeightPx(m, roundToInt(d));

        int y = 0;
        int x = 0;
        mixKnob.setBounds(x, y, cellWPx, cellH);    x += cellWPx + gapPx;
        gainKnob.setBounds(x, y, cellWPx, cellH);   x += cellWPx + gapPx;
        bitsKnob.setBounds(x, y, cellWPx, cellH);
    }

    void setAccentColour(Colour c) override
    {
        mixKnob.setAccentColour(c);
        gainKnob.setAccentColour(c);
        bitsKnob.setAccentColour(c);
    }

    void setMetrics(const Metrics& m) override
    {
        metrics_ = m;
        mixKnob.setMetrics(m);
        gainKnob.setMetrics(m);
        bitsKnob.setMetrics(m);
        resized();
    }

private:
    Metrics metrics_;
    PDLKnob mixKnob;
    PDLKnob gainKnob;
    PDLKnob bitsKnob;
};

} // namespace

ChronosEditor::ChronosEditor(ChronosProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p), tapDisplay_(p), header_(p), footer_(p),
      tooltipWindow_(this, kTooltipDelayMs)
{
    setLookAndFeel(&lnf_);

    addAndMakeVisible(header_);
    addAndMakeVisible(footer_);
    addAndMakeVisible(tapDisplay_);

    timeCard_.setContent(std::make_unique<TimePanel>(processorRef, knobLnf_));
    addAndMakeVisible(timeCard_);

    repeatsCard_.setContent(std::make_unique<RepeatsPanel>(processorRef, knobLnf_));
    addAndMakeVisible(repeatsCard_);

    diffuserCard_.setContent(std::make_unique<DiffuserPanel>(processorRef, knobLnf_));
    addAndMakeVisible(diffuserCard_);

    driveCard_.setContent(std::make_unique<DrivePanel>(processorRef, knobLnf_));
    addAndMakeVisible(driveCard_);

    filterCard_.setContent(std::make_unique<FilterPanel>(processorRef, knobLnf_));
    addAndMakeVisible(filterCard_);

    levelCard_.setContent(std::make_unique<LevelPanel>(processorRef, knobLnf_));
    addAndMakeVisible(levelCard_);

    const auto rawMode = processorRef.getParameters().getRawDelayMode();
    updateCoreAccentColour_(static_cast<float>(rawMode));
    pendingDelayMode_.store(rawMode, std::memory_order_relaxed);
    lastDelayMode_ = rawMode;
    const bool rawBypass = processorRef.getParameters().getBypass();
    pendingBypass_.store(rawBypass ? 1 : 0, std::memory_order_relaxed);
    lastBypass_ = rawBypass;

    processorRef.getAPVTS().addParameterListener(delayModeParamID.getParamID(), this);
    processorRef.getAPVTS().addParameterListener(bypassParamID.getParamID(), this);
    processorRef.getAPVTS().addParameterListener(delaySyncParamID.getParamID(), this);
    processorRef.getAPVTS().addParameterListener(timeLinkParamID.getParamID(), this);
    processorRef.getAPVTS().addParameterListener(enableDiffuserParamID.getParamID(), this);
    processorRef.getAPVTS().addParameterListener(adaaOrderParamID.getParamID(), this);

    paramPoll_ = std::make_unique<LambdaTimer>([this] { pollParameterChanges_(); }, 10);

    setResizable(true, true);
    setResizeLimits(Metrics::kMinWidth, Metrics::kMinHeight,
                    Metrics::kMaxWidth, Metrics::kMaxHeight);
    getConstrainer()->setFixedAspectRatio(Metrics::kDesignAspect);

    // Read the stored width only when the layout revision matches. A
    // session from an older layout opens at the default width.
    const int layoutRev = processorRef.getEditorLayoutRev();
    const int storedW = (layoutRev == 7) ? processorRef.getEditorWidth() : Metrics::kDefaultWidth;
    const int w = std::clamp(storedW, Metrics::kMinWidth, Metrics::kMaxWidth);
    const int h = juce::roundToInt(static_cast<float>(w) / static_cast<float>(Metrics::kDesignAspect));
    setSize(w, h);

    updateEnablement_();

    header_.setExplicitFocusOrder(1);
    tapDisplay_.setExplicitFocusOrder(2);
    timeCard_.setExplicitFocusOrder(3);
    repeatsCard_.setExplicitFocusOrder(4);
    diffuserCard_.setExplicitFocusOrder(5);
    driveCard_.setExplicitFocusOrder(6);
    filterCard_.setExplicitFocusOrder(7);
    levelCard_.setExplicitFocusOrder(8);
    footer_.setExplicitFocusOrder(9);

    tapDisplay_.setTitle("Tap Display");
    tapDisplay_.setTooltip("Drag the plot to set the delay time. Double-click to reset.");
    tapDisplay_.setHelpText("Drag the plot to set the delay time. Double-click to reset.");
    timeCard_.setTitle("Time");
    repeatsCard_.setTitle("Repeats");
    diffuserCard_.setTitle("Diffuser");
    driveCard_.setTitle("Drive");
    filterCard_.setTitle("Filter");
    levelCard_.setTitle("Level");
    footer_.setTitle("Footer");
}

ChronosEditor::~ChronosEditor()
{
    processorRef.setEditorOpen(false);
    paramPoll_.reset();
    stopTimer();
    processorRef.getAPVTS().removeParameterListener(delayModeParamID.getParamID(), this);
    processorRef.getAPVTS().removeParameterListener(bypassParamID.getParamID(), this);
    processorRef.getAPVTS().removeParameterListener(delaySyncParamID.getParamID(), this);
    processorRef.getAPVTS().removeParameterListener(timeLinkParamID.getParamID(), this);
    processorRef.getAPVTS().removeParameterListener(enableDiffuserParamID.getParamID(), this);
    processorRef.getAPVTS().removeParameterListener(adaaOrderParamID.getParamID(), this);
    setLookAndFeel(nullptr);
}

void ChronosEditor::parameterChanged(const String& parameterID, const float newValue)
{
    if (parameterID == delayModeParamID.getParamID())
        pendingDelayMode_.store(juce::roundToInt(newValue), std::memory_order_relaxed);
    else if (parameterID == bypassParamID.getParamID())
        pendingBypass_.store((juce::roundToInt(newValue) != 0) ? 1 : 0, std::memory_order_relaxed);

    if (parameterID == delaySyncParamID.getParamID()
        || parameterID == timeLinkParamID.getParamID()
        || parameterID == enableDiffuserParamID.getParamID()
        || parameterID == adaaOrderParamID.getParamID()
        || parameterID == bypassParamID.getParamID())
        triggerAsyncUpdate();
}

void ChronosEditor::handleAsyncUpdate()
{
    updateEnablement_();
}

void ChronosEditor::updateEnablement_()
{
    const auto& params = processorRef.getParameters();

    MarsDSP::GUI::EnablementState state;
    state.delaySync = params.getRawDelaySync();
    state.timeLink = params.getRawTimeLink();
    state.enableDiffuser = params.getRawEnableDiffuser();
    state.driveSatOff = (params.getADAAOrder() == 0);

    timeCard_.setEnablement(state);
    repeatsCard_.setEnablement(state);
    diffuserCard_.setEnablement(state);
    driveCard_.setEnablement(state);
    filterCard_.setEnablement(state);
    levelCard_.setEnablement(state);

    const bool live = ! params.getBypass();
    tapDisplay_.setEnabled(live);
    timeCard_.setEnabled(live);
    repeatsCard_.setEnabled(live);
    diffuserCard_.setEnabled(live);
    driveCard_.setEnabled(live);
    filterCard_.setEnabled(live);
    levelCard_.setEnabled(live);
}

void ChronosEditor::updateCoreAccentColour_(const float delayModeVal)
{
    const int mode = (delayModeVal > 0.5f) ? 1 : 0;
    const auto col = (mode == 1) ? MarsDSP::GUI::Colours::accentDelayBBD
                                 : MarsDSP::GUI::Colours::accentDelayDigital;
    tapDisplay_.setAccentColour(col);
    header_.setAccentColour(col);
    timeCard_.setAccentColour(col);
    repeatsCard_.setAccentColour(col);
    diffuserCard_.setAccentColour(col);
    driveCard_.setAccentColour(col);
    filterCard_.setAccentColour(col);
    levelCard_.setAccentColour(col);
}

void ChronosEditor::pollParameterChanges_()
{
    const int mode = pendingDelayMode_.load(std::memory_order_relaxed);
    if (mode != lastDelayMode_)
    {
        lastDelayMode_ = mode;
        updateCoreAccentColour_(static_cast<float>(mode));
    }

    const bool bp = (pendingBypass_.load(std::memory_order_relaxed) != 0);
    if (bp != lastBypass_)
    {
        lastBypass_ = bp;
        repaint();
    }
}

void ChronosEditor::paint(Graphics& g)
{
    g.fillAll(MarsDSP::GUI::Colours::background);
}

void ChronosEditor::paintOverChildren(Graphics& g)
{
    if (! processorRef.getParameters().getBypass())
        return;

    g.setColour(MarsDSP::GUI::Colours::background.withAlpha(MarsDSP::GUI::kBypassScrimAlpha));
    g.fillRect(tapDisplay_.getBounds());
    g.fillRect(cardAreaBounds_);
}

void ChronosEditor::resized()
{
    const auto m = Metrics::fromWidth(getWidth());
    metrics_ = m;
    lnf_.setMetrics(m);
    knobLnf_.setMetrics(m);

    header_.setMetrics(m);
    footer_.setMetrics(m);
    tapDisplay_.setMetrics(m);
    timeCard_.setMetrics(m);
    repeatsCard_.setMetrics(m);
    diffuserCard_.setMetrics(m);
    driveCard_.setMetrics(m);
    filterCard_.setMetrics(m);
    levelCard_.setMetrics(m);

    const int w = getWidth();
    const int side = m.px(Metrics::kSideMargin);
    const int gutter = m.px(Metrics::kCardGutter);

    int y = m.px(Metrics::kTopPad);
    const int headerH = m.px(Metrics::kHeaderH);
    header_.setBounds(0, y, w, headerH);
    y += headerH + m.px(Metrics::kGapHeader);

    const int tapH = m.px(Metrics::kTapH);
    tapDisplay_.setBounds(side, y, w - 2 * side, tapH);
    y += tapH + m.px(Metrics::kGapTap);

    // The two-column grid.
    const int colW = (w - 2 * side - gutter) / 2;
    const int row1H = m.px(Metrics::kRow1H);
    const int row2H = m.px(Metrics::kRow2H);

    const int cardY = y;
    const int leftX = side;
    const int rightX = side + colW + gutter;

    // Row 1: TIME and REPEATS.
    const int timeH = m.px(Metrics::kTimeCardH);
    timeCard_.setBounds(leftX, cardY, colW, timeH);
    repeatsCard_.setBounds(rightX, cardY, colW, row1H);

    // Row 2: DIFFUSER and the DRIVE/FILTER/LEVEL stack.
    const int row2Y = cardY + row1H + gutter;
    diffuserCard_.setBounds(leftX, row2Y, colW, row2H);

    int stackY = row2Y;
    const int driveH = m.px(Metrics::kDriveCardH);
    driveCard_.setBounds(rightX, stackY, colW, driveH);
    stackY += driveH + gutter;

    const int filterH = m.px(Metrics::kFilterCardH);
    filterCard_.setBounds(rightX, stackY, colW, filterH);
    stackY += filterH + gutter;

    const int levelH = m.px(Metrics::kLevelCardH);
    levelCard_.setBounds(rightX, stackY, colW, levelH);

    // The bypass scrim covers the full card area.
    cardAreaBounds_ = Rectangle<int>(side, cardY, w - 2 * side, m.px(Metrics::kCardAreaH));

    y = row2Y + row2H + m.px(Metrics::kGapCards);

    const int footerH = m.px(Metrics::kFooterH);
    footer_.setBounds(0, y, w, footerH);

    startTimer(250);
}

void ChronosEditor::timerCallback()
{
    stopTimer();

    // One write per settle, into the side tree. The parameter tree
    // never carries window geometry.
    processorRef.setEditorWidth(getWidth());
    processorRef.setEditorLayoutRev(7);
}
