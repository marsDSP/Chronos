#include "ChronosEditor.h"
#include "gui/controls/SegmentButtons.h"

using Metrics = MarsDSP::GUI::Metrics;

namespace {

using namespace MarsDSP::GUI::Knobs;
using GUIColours = MarsDSP::GUI::Colours;
using MarsDSP::GUI::Metrics;
using MarsDSP::GUI::AccentConsumer;
// Uniform grid constants (design units, section 4.4).

// Derive the knob diameter in pixels for n knobs in a content area.
// rowH is the pixel height available for the row (including label and readout space).
// hasReadout accounts for a value readout below the label.
// dMaxDU is the design-unit maximum diameter (58 for standard knobs, 72 for the hero).
float knobDiameterPx(const Metrics& m, const float contentW, const float rowH, const int n,
                    const bool hasReadout,
                    const float dMaxDU = static_cast<float>(Metrics::kKnobMax))
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
                      m.pxf(dMaxDU));
}

// Return the PDLKnob cell height in pixels: knob + label band + knob-to-label gap.
int knobCellHeightPx(const Metrics& m, const int dPx)
{
    return m.px(static_cast<float>(Metrics::kLabelBandH))
         + m.px(static_cast<float>(Metrics::kKnobLabelGap))
         + dPx;
}

// Return the x position of the first knob in a row of n knobs.
float knobRowStartX(const Metrics& m, const float contentW, const int n, const float dPx)
{
    const float g = m.pxf(static_cast<float>(Metrics::kKnobGutter));
    const float totalW = static_cast<float>(n) * dPx + static_cast<float>(n - 1) * g;
    return (contentW - totalW) * 0.5f;
}

// Return the core accent for the current delay mode.
Colour coreAccent(const ChronosProcessor& proc)
{
    return (proc.getParameters().getRawDelayMode() == 1) ? GUIColours::accentDelayBBD
                                                          : GUIColours::accentDelayDigital;
}

// 1. TIME card -> TIME sub-panel
class TimePanel final : public Component, public AccentConsumer {
public:
    explicit TimePanel(ChronosProcessor& proc)
        : timeLKnob("LEFT TIME", proc.getAPVTS(), delayTimeParamID),
          timeRKnob("RIGHT TIME", proc.getAPVTS(), delayTimeRParamID)
    {
        timeLDisplay.setSlider(&timeLKnob.getSlider());
        timeRDisplay.setSlider(&timeRKnob.getSlider());
        addAndMakeVisible(timeLDisplay);
        addAndMakeVisible(timeRDisplay);
        addAndMakeVisible(timeLKnob);
        addAndMakeVisible(timeRKnob);

        timeLinkButton.setColours(coreAccent(proc), GUIColours::textDim);
        timeLinkAttach = std::make_unique<AudioProcessorValueTreeState::ButtonAttachment>(
            proc.getAPVTS(), timeLinkParamID.getParamID(), timeLinkButton);
        addAndMakeVisible(timeLinkButton);

        syncButton.setMusicalNote(true);
        syncButton.setColours(coreAccent(proc), GUIColours::textDim);
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
        divisionAttach = std::make_unique<AudioProcessorValueTreeState::ComboBoxAttachment>(
            proc.getAPVTS(), delayDivisionParamID.getParamID(), divisionBox);
        addAndMakeVisible(divisionBox);
    }

    void resized() override
    {
        const auto m = MarsDSP::GUI::currentMetrics();
        const auto w = static_cast<float>(getWidth());
        const auto h = static_cast<float>(getHeight());

        const float g = m.pxf(static_cast<float>(Metrics::kKnobGutter));
        const int selH = m.px(Metrics::kSelectorRowH);
        const int rowGap = m.px(static_cast<float>(Metrics::kInterRowGap));
        const int readoutH = m.px(static_cast<float>(Metrics::kReadoutBandH));
        const int labelReadoutGap = m.px(static_cast<float>(Metrics::kLabelReadoutGap));

        // Row 1: two knobs with readouts. Row 2: the selector row.
        const float row1H = h - static_cast<float>(selH) - static_cast<float>(rowGap);
        const float d = knobDiameterPx(m, w, row1H, 2, true);
        const int dPx = roundToInt(d);
        const int cellH = knobCellHeightPx(m, dPx);
        const int cellWPx = roundToInt((w - g) / 2.0f);
        const int gapPx = roundToInt(g);

        // Centre the content block vertically.
        const int blockH = cellH + labelReadoutGap + readoutH + rowGap + selH;
        int y = roundToInt((h - static_cast<float>(blockH)) * 0.5f);

        int x = 0;
        timeLKnob.setBounds(x, y, cellWPx, cellH);
        timeRKnob.setBounds(x + cellWPx + gapPx, y, cellWPx, cellH);

        y += cellH + labelReadoutGap;
        timeLDisplay.setBounds(x, y, cellWPx, readoutH);
        timeRDisplay.setBounds(x + cellWPx + gapPx, y, cellWPx, readoutH);

        y += readoutH + rowGap;
        const int btnSize = m.px(24.0f);
        const int btnGap = m.px(4.0f);
        int sx = 0;
        timeLinkButton.setBounds(sx, y, btnSize, btnSize);  sx += btnSize + btnGap;
        syncButton.setBounds(sx, y, btnSize, btnSize);       sx += btnSize + btnGap;
        divisionBox.setBounds(sx, y + m.px(1.0f), getWidth() - sx, selH);
    }

    void setAccentColour(Colour c) override
    {
        timeLDisplay.setAccentColour(c);
        timeRDisplay.setAccentColour(c);
        timeLinkButton.setAccentColour(c);
        syncButton.setAccentColour(c);
    }

private:
    PDLKnob timeLKnob;
    PDLKnob timeRKnob;
    MarsDSP::GUI::TimeDisplay timeLDisplay;
    MarsDSP::GUI::TimeDisplay timeRDisplay;
    MarsDSP::GUI::TimeLockButton timeLinkButton;
    MarsDSP::GUI::PowerButton syncButton;
    ComboBox divisionBox;
    std::unique_ptr<AudioProcessorValueTreeState::ButtonAttachment> timeLinkAttach;
    std::unique_ptr<AudioProcessorValueTreeState::ButtonAttachment> syncAttach;
    std::unique_ptr<AudioProcessorValueTreeState::ComboBoxAttachment> divisionAttach;
};

// 2. TIME card -> MOD sub-panel
class ModPanel final : public Component {
public:
    explicit ModPanel(ChronosProcessor& proc)
        : depthKnob("MOD DEPTH", proc.getAPVTS(), delayModDepthParamID),
          rateKnob("MOD RATE", proc.getAPVTS(), delayModRateHzParamID)
    {
        addAndMakeVisible(depthKnob);
        addAndMakeVisible(rateKnob);
    }

    void resized() override
    {
        const auto m = MarsDSP::GUI::currentMetrics();
        const auto w = static_cast<float>(getWidth());
        const auto h = static_cast<float>(getHeight());

        const float g = m.pxf(static_cast<float>(Metrics::kKnobGutter));
        const float d = knobDiameterPx(m, w, h, 2, false);
        const int dPx = roundToInt(d);
        const int cellH = knobCellHeightPx(m, dPx);
        const int cellWPx = roundToInt((w - g) / 2.0f);
        const int gapPx = roundToInt(g);

        const int y = roundToInt((h - static_cast<float>(cellH)) * 0.5f);

        int x = 0;
        depthKnob.setBounds(x, y, cellWPx, cellH);
        rateKnob.setBounds(x + cellWPx + gapPx, y, cellWPx, cellH);
    }

private:
    PDLKnob depthKnob;
    PDLKnob rateKnob;
};

// 3. REPEATS card -> LOOP sub-panel
class LoopPanel final : public Component, public AccentConsumer {
public:
    explicit LoopPanel(ChronosProcessor& proc)
        : feedbackKnob("FEEDBACK", proc.getAPVTS(), feedbackParamID),
          crossFeedKnob("CROSS", proc.getAPVTS(), crossFeedParamID),
          loopDriveKnob("DRIVE", proc.getAPVTS(), loopDriveParamID),
          loopSatSeg_(proc.getAPVTS(), loopSatOrderParamID.getParamID(),
                      StringArray{"Off", "1st", "2nd"}, coreAccent(proc), true,
                      MarsDSP::GUI::kAntiAliasLabels, MarsDSP::GUI::kAntiAliasTooltips),
          delayModeSeg_(proc.getAPVTS(), delayModeParamID.getParamID(),
                        StringArray{"Digital", "BBD"}, coreAccent(proc), true)
    {
        addAndMakeVisible(feedbackKnob);
        addAndMakeVisible(crossFeedKnob);
        addAndMakeVisible(loopDriveKnob);
        addAndMakeVisible(loopSatSeg_);
        addAndMakeVisible(delayModeSeg_);
    }

    void resized() override
    {
        const auto m = MarsDSP::GUI::currentMetrics();
        const auto w = static_cast<float>(getWidth());
        const auto h = static_cast<float>(getHeight());

        const float g = m.pxf(static_cast<float>(Metrics::kKnobGutter));
        const int selH = m.px(Metrics::kSelectorRowH);
        const int rowGap = m.px(static_cast<float>(Metrics::kInterRowGap));

        // Row 1: the delay-mode segment. Row 2: three knobs. Row 3: the loop-sat segment.
        const float knobRowH = h - 2.0f * static_cast<float>(selH) - 2.0f * static_cast<float>(rowGap);
        const float d = knobDiameterPx(m, w, knobRowH, 3, false);
        const int dPx = roundToInt(d);
        const int cellH = knobCellHeightPx(m, dPx);
        const int cellWPx = roundToInt((w - 2.0f * g) / 3.0f);
        const int gapPx = roundToInt(g);

        const int blockH = selH + rowGap + cellH + rowGap + selH;
        int y = roundToInt((h - static_cast<float>(blockH)) * 0.5f);

        delayModeSeg_.setBounds(0, y, getWidth(), selH);
        y += selH + rowGap;

        int x = 0;
        feedbackKnob.setBounds(x, y, cellWPx, cellH);   x += cellWPx + gapPx;
        crossFeedKnob.setBounds(x, y, cellWPx, cellH);   x += cellWPx + gapPx;
        loopDriveKnob.setBounds(x, y, cellWPx, cellH);

        y += cellH + rowGap;
        loopSatSeg_.setBounds(0, y, getWidth(), selH);
    }

    void setAccentColour(Colour c) override
    {
        loopSatSeg_.setAccentColour(c);
        delayModeSeg_.setAccentColour(c);
    }

private:
    PDLKnob feedbackKnob;
    PDLKnob crossFeedKnob;
    PDLKnob loopDriveKnob;
    MarsDSP::GUI::SegmentButtons loopSatSeg_;
    MarsDSP::GUI::SegmentButtons delayModeSeg_;
};

// 4. REPEATS card -> TONE sub-panel
class TonePanel final : public Component {
public:
    explicit TonePanel(ChronosProcessor& proc)
        : dampKnob("DAMP", proc.getAPVTS(), dampHzParamID),
          loopCutKnob("CUT", proc.getAPVTS(), loopCutHzParamID)
    {
        addAndMakeVisible(dampKnob);
        addAndMakeVisible(loopCutKnob);
    }

    void resized() override
    {
        const auto m = MarsDSP::GUI::currentMetrics();
        const auto w = static_cast<float>(getWidth());
        const auto h = static_cast<float>(getHeight());

        const float g = m.pxf(static_cast<float>(Metrics::kKnobGutter));
        const float d = knobDiameterPx(m, w, h, 2, false);
        const int dPx = roundToInt(d);
        const int cellH = knobCellHeightPx(m, dPx);
        const int cellWPx = roundToInt((w - g) / 2.0f);
        const int gapPx = roundToInt(g);

        const int y = roundToInt((h - static_cast<float>(cellH)) * 0.5f);

        int x = 0;
        dampKnob.setBounds(x, y, cellWPx, cellH);
        loopCutKnob.setBounds(x + cellWPx + gapPx, y, cellWPx, cellH);
    }

private:
    PDLKnob dampKnob;
    PDLKnob loopCutKnob;
};

// 5. CHARACTER card -> DRIVE sub-panel
class DrivePanel final : public Component, public AccentConsumer {
public:
    explicit DrivePanel(ChronosProcessor& proc)
        : driveKnob("DRIVE", proc.getAPVTS(), driveParamID),
          bitsKnob("BIT DEPTH", proc.getAPVTS(), bitsParamID),
          adaaSeg_(proc.getAPVTS(), adaaOrderParamID.getParamID(),
                   StringArray{"Off", "1st", "2nd"}, coreAccent(proc), false,
                   MarsDSP::GUI::kAntiAliasLabels, MarsDSP::GUI::kAntiAliasTooltips)
    {
        addAndMakeVisible(driveKnob);
        addAndMakeVisible(bitsKnob);
        addAndMakeVisible(adaaSeg_);
    }

    void resized() override
    {
        const auto m = MarsDSP::GUI::currentMetrics();
        const auto w = static_cast<float>(getWidth());
        const auto h = static_cast<float>(getHeight());

        const int selH = m.px(Metrics::kSelectorRowH);
        const int rowGap = m.px(static_cast<float>(Metrics::kInterRowGap));
        const float g = m.pxf(static_cast<float>(Metrics::kKnobGutter));

        // Row 1: the hero drive knob and the bit-depth knob. Row 2: the anti-alias segment.
        const float row1H = h - static_cast<float>(selH) - static_cast<float>(rowGap);
        const float dDrive = knobDiameterPx(m, w, row1H, 2, false,
                                            static_cast<float>(Metrics::kHeroKnobMax));
        const float dBits  = knobDiameterPx(m, w, row1H, 2, false,
                                            static_cast<float>(Metrics::kKnobMax));
        const int cellHDrive = knobCellHeightPx(m, roundToInt(dDrive));
        const int cellHBits  = knobCellHeightPx(m, roundToInt(dBits));
        const int rowH = std::max(cellHDrive, cellHBits);
        const int cellWPx = roundToInt((w - g) / 2.0f);
        const int gapPx = roundToInt(g);

        const int blockH = rowH + rowGap + selH;
        int y = roundToInt((h - static_cast<float>(blockH)) * 0.5f);

        driveKnob.setBounds(0, y, cellWPx, cellHDrive);
        bitsKnob.setBounds(cellWPx + gapPx, y, cellWPx, cellHBits);

        y += rowH + rowGap;
        adaaSeg_.setBounds(0, y, getWidth(), selH);
    }

    void setAccentColour(Colour c) override
    {
        adaaSeg_.setAccentColour(c);
    }

private:
    PDLKnob driveKnob;
    PDLKnob bitsKnob;
    MarsDSP::GUI::SegmentButtons adaaSeg_;
};

// 6. CHARACTER card -> DIFFUSE sub-panel
class DiffusePanel final : public Component, public AccentConsumer {
public:
    explicit DiffusePanel(ChronosProcessor& proc)
        : diffusionKnob("DIFFUSION", proc.getAPVTS(), diffusionParamID),
          sizeKnob("SIZE", proc.getAPVTS(), diffuserSizeParamID),
          modDepthKnob("DIFF MOD", proc.getAPVTS(), diffModDepthParamID),
          modRateKnob("DIFF RATE", proc.getAPVTS(), diffModRateHzParamID)
    {
        enableButton.setColours(coreAccent(proc), GUIColours::textDim);
        enableAttach = std::make_unique<AudioProcessorValueTreeState::ButtonAttachment>(
            proc.getAPVTS(), enableDiffuserParamID.getParamID(), enableButton);
        addAndMakeVisible(enableButton);

        addAndMakeVisible(diffusionKnob);
        addAndMakeVisible(sizeKnob);
        addAndMakeVisible(modDepthKnob);
        addAndMakeVisible(modRateKnob);
    }

    void resized() override
    {
        const auto m = MarsDSP::GUI::currentMetrics();
        const auto w = static_cast<float>(getWidth());
        const auto h = static_cast<float>(getHeight());

        const float g = m.pxf(static_cast<float>(Metrics::kKnobGutter));
        const int rowGap = m.px(static_cast<float>(Metrics::kInterRowGap));
        const int enableH = m.px(Metrics::kEnableRowH);
        const int enableGap = m.px(8.0f);

        // Row 0: the enable button. Rows 1 and 2: the two knob rows.
        const float knobH = h - static_cast<float>(enableH) - static_cast<float>(enableGap)
                           - 2.0f * static_cast<float>(rowGap);
        const float rowH = (knobH - static_cast<float>(rowGap)) * 0.5f;
        const float d = knobDiameterPx(m, w, rowH, 2, false);
        const int dPx = roundToInt(d);
        const int cellH = knobCellHeightPx(m, dPx);
        const int cellWPx = roundToInt((w - g) / 2.0f);
        const int gapPx = roundToInt(g);

        const int blockH = enableH + enableGap + cellH + rowGap + cellH;
        int y = roundToInt((h - static_cast<float>(blockH)) * 0.5f);

        const int btnSize = m.px(24.0f);
        enableButton.setBounds(getWidth() - btnSize, y, btnSize, btnSize);
        y += enableH + enableGap;

        int x = 0;
        diffusionKnob.setBounds(x, y, cellWPx, cellH);  x += cellWPx + gapPx;
        sizeKnob.setBounds(x, y, cellWPx, cellH);

        y += cellH + rowGap;
        x = 0;
        modDepthKnob.setBounds(x, y, cellWPx, cellH);  x += cellWPx + gapPx;
        modRateKnob.setBounds(x, y, cellWPx, cellH);
    }

    void setAccentColour(Colour c) override
    {
        enableButton.setAccentColour(c);
    }

private:
    MarsDSP::GUI::PowerButton enableButton;
    PDLKnob diffusionKnob;
    PDLKnob sizeKnob;
    PDLKnob modDepthKnob;
    PDLKnob modRateKnob;
    std::unique_ptr<AudioProcessorValueTreeState::ButtonAttachment> enableAttach;
};

// 7. OUTPUT card -> FILTER sub-panel
class FilterPanel final : public Component, public AccentConsumer {
public:
    explicit FilterPanel(ChronosProcessor& proc)
        : hpfKnob("OUTPUT HPF", proc.getAPVTS(), hpfFreqParamID),
          lpfKnob("OUTPUT LPF", proc.getAPVTS(), lpfFreqParamID),
          modeSeg_(proc.getAPVTS(), filterModeParamID.getParamID(),
                   StringArray{"Digital", "Analog"}, coreAccent(proc), false)
    {
        addAndMakeVisible(modeSeg_);
        addAndMakeVisible(hpfKnob);
        addAndMakeVisible(lpfKnob);
    }

    void resized() override
    {
        const auto m = MarsDSP::GUI::currentMetrics();
        const auto w = static_cast<float>(getWidth());
        const auto h = static_cast<float>(getHeight());

        const float g = m.pxf(static_cast<float>(Metrics::kKnobGutter));
        const int selH = m.px(Metrics::kSelectorRowH);
        const int rowGap = m.px(static_cast<float>(Metrics::kInterRowGap));

        // Row 1: the filter-mode segment. Row 2: two knobs.
        const float row2H = h - static_cast<float>(selH) - static_cast<float>(rowGap);
        const float d = knobDiameterPx(m, w, row2H, 2, false);
        const int dPx = roundToInt(d);
        const int cellH = knobCellHeightPx(m, dPx);
        const int cellWPx = roundToInt((w - g) / 2.0f);
        const int gapPx = roundToInt(g);

        const int blockH = selH + rowGap + cellH;
        int y = roundToInt((h - static_cast<float>(blockH)) * 0.5f);

        modeSeg_.setBounds(0, y, getWidth(), selH);

        y += selH + rowGap;
        int x = 0;
        hpfKnob.setBounds(x, y, cellWPx, cellH);  x += cellWPx + gapPx;
        lpfKnob.setBounds(x, y, cellWPx, cellH);
    }

    void setAccentColour(Colour c) override
    {
        modeSeg_.setAccentColour(c);
    }

private:
    PDLKnob hpfKnob;
    PDLKnob lpfKnob;
    MarsDSP::GUI::SegmentButtons modeSeg_;
};

// 8. OUTPUT card -> LEVEL sub-panel
class LevelPanel final : public Component {
public:
    explicit LevelPanel(ChronosProcessor& proc)
        : mixKnob("MIX", proc.getAPVTS(), mixParamID),
          gainKnob("GAIN", proc.getAPVTS(), gainParamID)
    {
        addAndMakeVisible(mixKnob);
        addAndMakeVisible(gainKnob);
    }

    void resized() override
    {
        const auto m = MarsDSP::GUI::currentMetrics();
        const auto w = static_cast<float>(getWidth());
        const auto h = static_cast<float>(getHeight());

        const float g = m.pxf(static_cast<float>(Metrics::kKnobGutter));
        const float d = knobDiameterPx(m, w, h, 2, false);
        const int dPx = roundToInt(d);
        const int cellH = knobCellHeightPx(m, dPx);
        const int cellWPx = roundToInt((w - g) / 2.0f);
        const int gapPx = roundToInt(g);

        const int y = roundToInt((h - static_cast<float>(cellH)) * 0.5f);

        int x = 0;
        mixKnob.setBounds(x, y, cellWPx, cellH);   x += cellWPx + gapPx;
        gainKnob.setBounds(x, y, cellWPx, cellH);
    }

private:
    PDLKnob mixKnob;
    PDLKnob gainKnob;
};

} // namespace

ChronosEditor::ChronosEditor(ChronosProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p), tapDisplay_(p), header_(p), footer_(p)
{
    setLookAndFeel(&lnf_);

    addAndMakeVisible(header_);
    addAndMakeVisible(footer_);
    addAndMakeVisible(tapDisplay_);

    timeCard_.addContent("TIME", std::make_unique<TimePanel>(processorRef));
    timeCard_.addContent("MOD", std::make_unique<ModPanel>(processorRef));
    addAndMakeVisible(timeCard_);

    repeatsCard_.addContent("LOOP", std::make_unique<LoopPanel>(processorRef));
    repeatsCard_.addContent("TONE", std::make_unique<TonePanel>(processorRef));
    addAndMakeVisible(repeatsCard_);

    characterCard_.addContent("DRIVE", std::make_unique<DrivePanel>(processorRef));
    characterCard_.addContent("DIFFUSE", std::make_unique<DiffusePanel>(processorRef));
    addAndMakeVisible(characterCard_);

    outputCard_.addContent("FILTER", std::make_unique<FilterPanel>(processorRef));
    outputCard_.addContent("LEVEL", std::make_unique<LevelPanel>(processorRef));
    addAndMakeVisible(outputCard_);

    const auto rawMode = processorRef.getParameters().getRawDelayMode();
    updateCoreAccentColour_(static_cast<float>(rawMode));

    processorRef.getAPVTS().addParameterListener("delayMode", this);
    processorRef.getAPVTS().addParameterListener(bypassParamID.getParamID(), this);

    setResizable(true, true);
    setResizeLimits(Metrics::kMinWidth, Metrics::kMinHeight,
                    Metrics::kMaxWidth, Metrics::kMaxHeight);
    getConstrainer()->setFixedAspectRatio(Metrics::kDesignAspect);
    setSize(Metrics::kDefaultWidth, Metrics::kDefaultHeight);
}

ChronosEditor::~ChronosEditor()
{
    processorRef.getAPVTS().removeParameterListener("delayMode", this);
    processorRef.getAPVTS().removeParameterListener(bypassParamID.getParamID(), this);
    setLookAndFeel(nullptr);
}

void ChronosEditor::parameterChanged(const String& parameterID, const float newValue)
{
    const auto safe = SafePointer(this);

    if (parameterID == "delayMode")
    {
        MessageManager::callAsync([safe, newValue]
        {
            if (safe != nullptr)
                safe->updateCoreAccentColour_(newValue);
        });
    }
    else if (parameterID == bypassParamID.getParamID())
    {
        MessageManager::callAsync([safe]
        {
            if (safe != nullptr)
                safe->repaint();
        });
    }
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
    characterCard_.setAccentColour(col);
    outputCard_.setAccentColour(col);
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

    const auto cardRow = Rectangle<int>(timeCard_.getX(),
                                        timeCard_.getY(),
                                        outputCard_.getRight() - timeCard_.getX(),
                                        timeCard_.getHeight());
    g.fillRect(cardRow);
}

void ChronosEditor::resized()
{
    const auto m = Metrics::fromWidth(getWidth());
    metrics_ = m;
    setCurrentMetrics(m);

    header_.setMetrics(m);
    footer_.setMetrics(m);
    tapDisplay_.setMetrics(m);
    timeCard_.setMetrics(m);
    repeatsCard_.setMetrics(m);
    characterCard_.setMetrics(m);
    outputCard_.setMetrics(m);

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

    const int cardH = m.px(Metrics::kCardRowH);
    const int cardW = (w - 2 * side - 3 * gutter) / 4;
    int cx = side;
    timeCard_.setBounds(cx, y, cardW, cardH);      cx += cardW + gutter;
    repeatsCard_.setBounds(cx, y, cardW, cardH);    cx += cardW + gutter;
    characterCard_.setBounds(cx, y, cardW, cardH);  cx += cardW + gutter;
    outputCard_.setBounds(cx, y, cardW, cardH);
    y += cardH + m.px(Metrics::kGapCards);

    const int footerH = m.px(Metrics::kFooterH);
    footer_.setBounds(0, y, w, footerH);
}
