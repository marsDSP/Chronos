#include "ChronosEditor.h"

namespace {

using namespace MarsDSP::GUI::Knobs;
using GUIColours = MarsDSP::GUI::Colours;

// 1. TIME card -> TIME sub-panel
class TimePanel final : public Component {
public:
    explicit TimePanel(ChronosProcessor& proc)
        : timeLKnob("LEFT TIME", proc.getAPVTS(), delayTimeParamID, GUIColours::accentDelayDigital),
          timeRKnob("RIGHT TIME", proc.getAPVTS(), delayTimeRParamID, GUIColours::accentDelayDigital)
    {
        timeLDisplay.setSlider(&timeLKnob.getSlider());
        timeRDisplay.setSlider(&timeRKnob.getSlider());
        addAndMakeVisible(timeLDisplay);
        addAndMakeVisible(timeRDisplay);
        addAndMakeVisible(timeLKnob);
        addAndMakeVisible(timeRKnob);

        timeLinkButton.setColours(GUIColours::accentDelayDigital, GUIColours::textDim);
        timeLinkAttach = std::make_unique<AudioProcessorValueTreeState::ButtonAttachment>(
            proc.getAPVTS(), timeLinkParamID.getParamID(), timeLinkButton);
        addAndMakeVisible(timeLinkButton);

        syncButton.setMusicalNote(true);
        syncButton.setColours(GUIColours::accentDelayDigital, GUIColours::textDim);
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
        constexpr int knobSize = 76;
        constexpr int knobGap = 16;
        const int x0 = 10;
        const int y0 = 8;

        timeLKnob.setBounds(x0, y0, knobSize, knobSize);
        timeRKnob.setBounds(x0 + knobSize + knobGap, y0, knobSize, knobSize);

        timeLDisplay.setBounds(x0, y0 + knobSize + 2, knobSize, 20);
        timeRDisplay.setBounds(x0 + knobSize + knobGap, y0 + knobSize + 2, knobSize, 20);

        const int yCtrl = y0 + knobSize + 28;
        timeLinkButton.setBounds(x0, yCtrl, 24, 24);
        syncButton.setBounds(x0 + 30, yCtrl, 24, 24);
        divisionBox.setBounds(x0 + 60, yCtrl + 1, 92, 22);
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
        constexpr int knobSize = 84;
        constexpr int knobGap = 14;
        const int x0 = 14;
        const int y0 = 30;

        depthKnob.setBounds(x0, y0, knobSize, knobSize + 16);
        rateKnob.setBounds(x0 + knobSize + knobGap, y0, knobSize, knobSize + 16);
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
          loopDriveKnob("DRIVE", proc.getAPVTS(), loopDriveParamID, GUIColours::accentOrange)
    {
        addAndMakeVisible(feedbackKnob);
        addAndMakeVisible(crossFeedKnob);
        addAndMakeVisible(loopDriveKnob);

        loopSatBox.addItemList(StringArray{"Off", "1st", "2nd"}, 1);
        loopSatAttach = std::make_unique<AudioProcessorValueTreeState::ComboBoxAttachment>(
            proc.getAPVTS(), loopSatOrderParamID.getParamID(), loopSatBox);
        addAndMakeVisible(loopSatBox);

        delayModeBox.addItemList(StringArray{"Digital", "BBD"}, 1);
        delayModeAttach = std::make_unique<AudioProcessorValueTreeState::ComboBoxAttachment>(
            proc.getAPVTS(), delayModeParamID.getParamID(), delayModeBox);
        addAndMakeVisible(delayModeBox);
    }

    void resized() override
    {
        constexpr int knobSize = 64;
        constexpr int knobGap = 8;
        const int x0 = 8;
        const int y0 = 6;

        int x = x0;
        feedbackKnob.setBounds(x, y0, knobSize, knobSize + 12);  x += knobSize + knobGap;
        crossFeedKnob.setBounds(x, y0, knobSize, knobSize + 12); x += knobSize + knobGap;
        loopDriveKnob.setBounds(x, y0, knobSize, knobSize + 12);

        const int ySel = y0 + knobSize + 20;
        loopSatBox.setBounds(x0, ySel, 92, 22);
        delayModeBox.setBounds(x0 + 92 + 8, ySel, 92, 22);
    }

private:
    PDLKnob feedbackKnob;
    PDLKnob crossFeedKnob;
    PDLKnob loopDriveKnob;
    ComboBox loopSatBox;
    ComboBox delayModeBox;
    std::unique_ptr<AudioProcessorValueTreeState::ComboBoxAttachment> loopSatAttach;
    std::unique_ptr<AudioProcessorValueTreeState::ComboBoxAttachment> delayModeAttach;
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
        constexpr int knobSize = 84;
        constexpr int knobGap = 14;
        const int x0 = 14;
        const int y0 = 30;

        dampKnob.setBounds(x0, y0, knobSize, knobSize + 16);
        loopCutKnob.setBounds(x0 + knobSize + knobGap, y0, knobSize, knobSize + 16);
    }

private:
    PDLKnob dampKnob;
    PDLKnob loopCutKnob;
};

// 5. CHARACTER card -> DRIVE sub-panel
class DrivePanel final : public Component {
public:
    explicit DrivePanel(ChronosProcessor& proc)
        : driveKnob("DRIVE", proc.getAPVTS(), driveParamID, GUIColours::accentRed)
    {
        addAndMakeVisible(driveKnob);

        adaaOrderBox.addItemList(StringArray{"Off", "1st", "2nd"}, 1);
        adaaAttach = std::make_unique<AudioProcessorValueTreeState::ComboBoxAttachment>(
            proc.getAPVTS(), adaaOrderParamID.getParamID(), adaaOrderBox);
        addAndMakeVisible(adaaOrderBox);
    }

    void resized() override
    {
        constexpr int knobSize = 84;
        const int x0 = 14;
        const int y0 = 30;

        driveKnob.setBounds(x0, y0, knobSize, knobSize + 16);
        adaaOrderBox.setBounds(x0, y0 + knobSize + 22, 120, 22);
    }

private:
    PDLKnob driveKnob;
    ComboBox adaaOrderBox;
    std::unique_ptr<AudioProcessorValueTreeState::ComboBoxAttachment> adaaAttach;
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
        constexpr int knobSize = 72;
        constexpr int knobGap = 8;
        const int x0 = 14;

        enableButton.setBounds(getWidth() / 2 - 12, 6, 24, 24);

        const int y1 = 36;
        diffusionKnob.setBounds(x0, y1, knobSize, knobSize + 12);
        sizeKnob.setBounds(x0 + knobSize + knobGap, y1, knobSize, knobSize + 12);

        const int y2 = y1 + knobSize + 16;
        modDepthKnob.setBounds(x0, y2, knobSize, knobSize + 12);
        modRateKnob.setBounds(x0 + knobSize + knobGap, y2, knobSize, knobSize + 12);
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
          lpfKnob("OUTPUT LPF", proc.getAPVTS(), lpfFreqParamID, GUIColours::accentBlue)
    {
        filterModeBox.addItemList(StringArray{"Digital", "Analog"}, 1);
        modeAttach = std::make_unique<AudioProcessorValueTreeState::ComboBoxAttachment>(
            proc.getAPVTS(), filterModeParamID.getParamID(), filterModeBox);
        addAndMakeVisible(filterModeBox);

        addAndMakeVisible(hpfKnob);
        addAndMakeVisible(lpfKnob);
    }

    void resized() override
    {
        constexpr int knobSize = 84;
        constexpr int knobGap = 14;
        const int x0 = 14;

        filterModeBox.setBounds(x0, 8, 120, 22);

        const int y0 = 38;
        hpfKnob.setBounds(x0, y0, knobSize, knobSize + 16);
        lpfKnob.setBounds(x0 + knobSize + knobGap, y0, knobSize, knobSize + 16);
    }

private:
    ComboBox filterModeBox;
    PDLKnob hpfKnob;
    PDLKnob lpfKnob;
    std::unique_ptr<AudioProcessorValueTreeState::ComboBoxAttachment> modeAttach;
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
        constexpr int knobSize = 64;
        constexpr int knobGap = 8;
        const int x0 = 8;
        const int y0 = 20;

        int x = x0;
        mixKnob.setBounds(x, y0, knobSize, knobSize + 12);  x += knobSize + knobGap;
        gainKnob.setBounds(x, y0, knobSize, knobSize + 12); x += knobSize + knobGap;
        bitsKnob.setBounds(x, y0, knobSize, knobSize + 12);
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
    setResizeLimits(600, 384, 1600, 1024);
    getConstrainer()->setFixedAspectRatio(1000.0 / 640.0);
    setSize(1000, 640);
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
        MessageManager::callAsync([this, newValue]
        {
            updateCoreAccentColour_(newValue);
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
    const int w = getWidth();
    const int h = getHeight();
    const float scale = static_cast<float>(h) / 640.0f;

    const int headerH = juce::roundToInt(52.0f * scale);
    const int footerH = juce::roundToInt(36.0f * scale);
    const int tapH = juce::roundToInt(200.0f * scale);

    header_.setBounds(0, 0, w, headerH);
    footer_.setBounds(0, h - footerH, w, footerH);

    const int tapY = headerH + 4;
    tapDisplay_.setBounds(12, tapY, w - 24, tapH);

    const int cardY = tapY + tapH + 8;
    const int cardH = h - cardY - footerH - 4;
    constexpr int cardInset = 12;
    constexpr int cardGap = 8;
    const int cardW = (w - 2 * cardInset - 3 * cardGap) / 4;
    int cx = cardInset;
    timeCard_.setBounds(cx, cardY, cardW, cardH);      cx += cardW + cardGap;
    repeatsCard_.setBounds(cx, cardY, cardW, cardH);    cx += cardW + cardGap;
    characterCard_.setBounds(cx, cardY, cardW, cardH);  cx += cardW + cardGap;
    outputCard_.setBounds(cx, cardY, cardW, cardH);
}
