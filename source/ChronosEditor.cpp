#include "ChronosEditor.h"
#include "gui/controls/SegmentButtons.h"

using Metrics = MarsDSP::GUI::Metrics;

namespace {

using namespace MarsDSP::GUI::Knobs;
using GUIColours = MarsDSP::GUI::Colours;

// Uniform grid constants.
constexpr int kPad = 12;
constexpr int kKnobGap = 10;
constexpr int kSelectorH = 22;
constexpr int kDisplayH = 18;

// Scale one knob cell width to fit the panel. Keep it inside a safe band.
int knobCellWidth(const int panelWidth, const int cols)
{
    const int w = (panelWidth - 2 * kPad - (cols - 1) * kKnobGap) / cols;
    return std::clamp(w, 40, 80);
}

// Return the core accent for the current delay mode.
Colour coreAccent(const ChronosProcessor& proc)
{
    return (proc.getParameters().getRawDelayMode() == 1) ? GUIColours::accentDelayBBD
                                                          : GUIColours::accentDelayDigital;
}

// 1. TIME card -> TIME sub-panel
class TimePanel final : public Component {
public:
    explicit TimePanel(ChronosProcessor& proc)
        : timeLKnob("LEFT TIME", proc.getAPVTS(), delayTimeParamID, coreAccent(proc)),
          timeRKnob("RIGHT TIME", proc.getAPVTS(), delayTimeRParamID, coreAccent(proc))
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
        const int w = getWidth();
        const int kw = knobCellWidth(w, 2);
        const int kh = kw + 14;
        int x = kPad;
        const int y = kPad;

        timeLKnob.setBounds(x, y, kw, kh);
        timeRKnob.setBounds(x + kw + kKnobGap, y, kw, kh);

        timeLDisplay.setBounds(x, y + kh + 2, kw, kDisplayH);
        timeRDisplay.setBounds(x + kw + kKnobGap, y + kh + 2, kw, kDisplayH);

        const int yc = y + kh + kDisplayH + 14;
        timeLinkButton.setBounds(x, yc, 24, 24);
        syncButton.setBounds(x + 28, yc, 24, 24);
        divisionBox.setBounds(x + 56, yc + 1, w - 2 * kPad - 56, kSelectorH);
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
        : depthKnob("MOD DEPTH", proc.getAPVTS(), delayModDepthParamID, GUIColours::accentYellow),
          rateKnob("MOD RATE", proc.getAPVTS(), delayModRateHzParamID, GUIColours::accentYellow)
    {
        addAndMakeVisible(depthKnob);
        addAndMakeVisible(rateKnob);
    }

    void resized() override
    {
        const int w = getWidth();
        const int kw = knobCellWidth(w, 2);
        const int kh = kw + 14;
        const int totalW = 2 * kw + kKnobGap;
        const int x0 = (w - totalW) / 2;
        const int y = kPad + 12;

        depthKnob.setBounds(x0, y, kw, kh);
        rateKnob.setBounds(x0 + kw + kKnobGap, y, kw, kh);
    }

private:
    PDLKnob depthKnob;
    PDLKnob rateKnob;
};

// 3. REPEATS card -> LOOP sub-panel
class LoopPanel final : public Component {
public:
    explicit LoopPanel(ChronosProcessor& proc)
        : feedbackKnob("FEEDBACK", proc.getAPVTS(), feedbackParamID, GUIColours::accentOrange),
          crossFeedKnob("CROSS", proc.getAPVTS(), crossFeedParamID, GUIColours::accentOrange),
          loopDriveKnob("DRIVE", proc.getAPVTS(), loopDriveParamID, GUIColours::accentOrange),
          loopSatSeg_(proc.getAPVTS(), loopSatOrderParamID.getParamID(),
                      StringArray{"Off", "1st", "2nd"}, coreAccent(proc), true),
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
        const int w = getWidth();
        const int kw = knobCellWidth(w, 3);
        const int kh = kw + 14;
        int x = kPad;
        const int y = kPad;

        feedbackKnob.setBounds(x, y, kw, kh);   x += kw + kKnobGap;
        crossFeedKnob.setBounds(x, y, kw, kh);  x += kw + kKnobGap;
        loopDriveKnob.setBounds(x, y, kw, kh);

        const int y1 = y + kh + 12;
        loopSatSeg_.setBounds(kPad, y1, w - 2 * kPad, kSelectorH);
        delayModeSeg_.setBounds(kPad, y1 + kSelectorH + 6, w - 2 * kPad, kSelectorH);
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
        : dampKnob("DAMP", proc.getAPVTS(), dampHzParamID, GUIColours::accentOrange),
          loopCutKnob("CUT", proc.getAPVTS(), loopCutHzParamID, GUIColours::accentOrange)
    {
        addAndMakeVisible(dampKnob);
        addAndMakeVisible(loopCutKnob);
    }

    void resized() override
    {
        const int w = getWidth();
        const int kw = knobCellWidth(w, 2);
        const int kh = kw + 14;
        const int totalW = 2 * kw + kKnobGap;
        const int x0 = (w - totalW) / 2;
        const int y = kPad + 12;

        dampKnob.setBounds(x0, y, kw, kh);
        loopCutKnob.setBounds(x0 + kw + kKnobGap, y, kw, kh);
    }

private:
    PDLKnob dampKnob;
    PDLKnob loopCutKnob;
};

// 5. CHARACTER card -> DRIVE sub-panel
class DrivePanel final : public Component {
public:
    explicit DrivePanel(ChronosProcessor& proc)
        : driveKnob("DRIVE", proc.getAPVTS(), driveParamID, GUIColours::accentRed),
          adaaSeg_(proc.getAPVTS(), adaaOrderParamID.getParamID(),
                   StringArray{"Off", "1st", "2nd"}, GUIColours::accentPurple, false)
    {
        addAndMakeVisible(driveKnob);
        addAndMakeVisible(adaaSeg_);
    }

    void resized() override
    {
        const int w = getWidth();
        const int kw = knobCellWidth(w, 1);
        const int kh = kw + 14;
        const int x0 = (w - kw) / 2;
        const int y = kPad + 8;

        driveKnob.setBounds(x0, y, kw, kh);
        adaaSeg_.setBounds(kPad, y + kh + 12, w - 2 * kPad, kSelectorH);
    }

private:
    PDLKnob driveKnob;
    MarsDSP::GUI::SegmentButtons adaaSeg_;
};

// 6. CHARACTER card -> DIFFUSE sub-panel
class DiffusePanel final : public Component {
public:
    explicit DiffusePanel(ChronosProcessor& proc)
        : diffusionKnob("DIFFUSION", proc.getAPVTS(), diffusionParamID, GUIColours::accentPurple),
          sizeKnob("SIZE", proc.getAPVTS(), diffuserSizeParamID, GUIColours::accentPurple),
          modDepthKnob("DIFF MOD", proc.getAPVTS(), diffModDepthParamID, GUIColours::accentPurple),
          modRateKnob("DIFF RATE", proc.getAPVTS(), diffModRateHzParamID, GUIColours::accentPurple)
    {
        enableButton.setColours(GUIColours::accentPurple, GUIColours::textDim);
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
        const int w = getWidth();
        const int kw = knobCellWidth(w, 2);
        const int kh = kw + 14;
        int x = kPad;

        enableButton.setBounds((w - 24) / 2, 4, 24, 24);

        const int y1 = kPad + 30;
        diffusionKnob.setBounds(x, y1, kw, kh);
        sizeKnob.setBounds(x + kw + kKnobGap, y1, kw, kh);

        const int y2 = y1 + kh + 8;
        modDepthKnob.setBounds(x, y2, kw, kh);
        modRateKnob.setBounds(x + kw + kKnobGap, y2, kw, kh);
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
class FilterPanel final : public Component {
public:
    explicit FilterPanel(ChronosProcessor& proc)
        : hpfKnob("OUTPUT HPF", proc.getAPVTS(), hpfFreqParamID, GUIColours::accentBlue),
          lpfKnob("OUTPUT LPF", proc.getAPVTS(), lpfFreqParamID, GUIColours::accentBlue),
          modeSeg_(proc.getAPVTS(), filterModeParamID.getParamID(),
                   StringArray{"Digital", "Analog"}, GUIColours::accentBlue, false)
    {
        addAndMakeVisible(modeSeg_);
        addAndMakeVisible(hpfKnob);
        addAndMakeVisible(lpfKnob);
    }

    void resized() override
    {
        const int w = getWidth();
        const int kw = knobCellWidth(w, 2);
        const int kh = kw + 14;
        const int totalW = 2 * kw + kKnobGap;
        const int x0 = (w - totalW) / 2;

        modeSeg_.setBounds(kPad, kPad, w - 2 * kPad, kSelectorH);

        const int y = kPad + kSelectorH + 14;
        hpfKnob.setBounds(x0, y, kw, kh);
        lpfKnob.setBounds(x0 + kw + kKnobGap, y, kw, kh);
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
        : mixKnob("MIX", proc.getAPVTS(), mixParamID, GUIColours::accentBlue),
          gainKnob("GAIN", proc.getAPVTS(), gainParamID, GUIColours::accentBlue),
          bitsKnob("BIT DEPTH", proc.getAPVTS(), bitsParamID, GUIColours::accentBlue)
    {
        addAndMakeVisible(mixKnob);
        addAndMakeVisible(gainKnob);
        addAndMakeVisible(bitsKnob);
    }

    void resized() override
    {
        const int w = getWidth();
        const int kw = knobCellWidth(w, 3);
        const int kh = kw + 14;
        int x = kPad;
        const int y = kPad + 16;

        mixKnob.setBounds(x, y, kw, kh);   x += kw + kKnobGap;
        gainKnob.setBounds(x, y, kw, kh);  x += kw + kKnobGap;
        bitsKnob.setBounds(x, y, kw, kh);
    }

private:
    PDLKnob mixKnob;
    PDLKnob gainKnob;
    PDLKnob bitsKnob;
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

    characterCard_.setAccentColour(MarsDSP::GUI::Colours::accentPurple);
    outputCard_.setAccentColour(MarsDSP::GUI::Colours::accentBlue);

    const auto rawMode = processorRef.getParameters().getRawDelayMode();
    updateCoreAccentColour_(static_cast<float>(rawMode));

    processorRef.getAPVTS().addParameterListener("delayMode", this);

    setResizable(true, true);
    setResizeLimits(Metrics::kMinWidth, Metrics::kMinHeight,
                    Metrics::kMaxWidth, Metrics::kMaxHeight);
    getConstrainer()->setFixedAspectRatio(Metrics::kDesignAspect);
    setSize(Metrics::kDefaultWidth, Metrics::kDefaultHeight);
}

ChronosEditor::~ChronosEditor()
{
    processorRef.getAPVTS().removeParameterListener("delayMode", this);
    setLookAndFeel(nullptr);
}

void ChronosEditor::parameterChanged(const String& parameterID, const float newValue)
{
    if (parameterID == "delayMode")
    {
        const auto safe = Component::SafePointer<ChronosEditor>(this);
        MessageManager::callAsync([safe, newValue]
        {
            if (safe != nullptr)
                safe->updateCoreAccentColour_(newValue);
        });
    }
}

void ChronosEditor::updateCoreAccentColour_(const float delayModeVal)
{
    const int mode = (delayModeVal > 0.5f) ? 1 : 0;
    const auto col = (mode == 1) ? MarsDSP::GUI::Colours::accentDelayBBD
                                 : MarsDSP::GUI::Colours::accentDelayDigital;
    timeCard_.setAccentColour(col);
    repeatsCard_.setAccentColour(col);
    header_.setCoreMode(mode, col);
}

void ChronosEditor::paint(Graphics& g)
{
    g.fillAll(MarsDSP::GUI::Colours::background);
}

void ChronosEditor::resized()
{
    const auto m = Metrics::fromWidth(getWidth());
    metrics_ = m;

    header_.setMetrics(m);
    footer_.setMetrics(m);
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
