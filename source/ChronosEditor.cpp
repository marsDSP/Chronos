#include "ChronosEditor.h"

namespace {

using namespace MarsDSP::GUI::Knobs;
using GUIColours = MarsDSP::GUI::Colours;

// 1. DELAY -> TAPS sub-panel
class TapsPanel final : public Component {
public:
    explicit TapsPanel(ChronosProcessor& proc)
        : processorRef(proc),
          tapDisplay(proc),
          timeLKnob("LEFT TIME", proc.getAPVTS(), delayTimeParamID, GUIColours::accentDelayDigital),
          timeRKnob("RIGHT TIME", proc.getAPVTS(), delayTimeRParamID, GUIColours::accentDelayDigital)
    {
        addAndMakeVisible(tapDisplay);

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
        auto bounds = getLocalBounds();
        const int tapHeight = std::min(170, bounds.getHeight() * 45 / 100);
        tapDisplay.setBounds(bounds.removeFromTop(tapHeight));

        bounds.removeFromTop(12);

        const int knobSize = 80;

        timeLKnob.setBounds(16, bounds.getY(), knobSize, knobSize);
        timeLDisplay.setBounds(16, timeLKnob.getBottom() + 4, knobSize, 22);

        timeLinkButton.setBounds(120, bounds.getY() + 10, 24, 24);
        syncButton.setBounds(120, bounds.getY() + 42, 24, 24);
        divisionBox.setBounds(108, bounds.getY() + 74, 52, 22);

        timeRKnob.setBounds(180, bounds.getY(), knobSize, knobSize);
        timeRDisplay.setBounds(180, timeRKnob.getBottom() + 4, knobSize, 22);
    }

private:
    ChronosProcessor& processorRef;
    MarsDSP::GUI::TapDisplay tapDisplay;
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

// 2. DELAY -> REPEATS sub-panel
class RepeatsPanel final : public Component {
public:
    explicit RepeatsPanel(ChronosProcessor& proc)
        : feedbackKnob("FEEDBACK", proc.getAPVTS(), feedbackParamID, GUIColours::accentOrange),
          dampKnob("DAMP", proc.getAPVTS(), dampHzParamID, GUIColours::accentOrange),
          loopCutKnob("CUT", proc.getAPVTS(), loopCutHzParamID, GUIColours::accentOrange),
          crossFeedKnob("CROSS", proc.getAPVTS(), crossFeedParamID, GUIColours::accentOrange),
          loopDriveKnob("DRIVE", proc.getAPVTS(), loopDriveParamID, GUIColours::accentOrange),
          delayModeButton("Delay Core")
    {
        addAndMakeVisible(feedbackKnob);
        addAndMakeVisible(dampKnob);
        addAndMakeVisible(loopCutKnob);
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
        auto bounds = getLocalBounds().reduced(8);
        constexpr int knobWidth = 85;
        constexpr int knobHeight = 90;
        constexpr int gap = 12;

        int x = bounds.getX();
        feedbackKnob.setBounds(x, bounds.getY() + 10, knobWidth, knobHeight);
        x += knobWidth + gap;
        dampKnob.setBounds(x, bounds.getY() + 10, knobWidth, knobHeight);
        x += knobWidth + gap;
        loopCutKnob.setBounds(x, bounds.getY() + 10, knobWidth, knobHeight);
        x += knobWidth + gap;
        crossFeedKnob.setBounds(x, bounds.getY() + 10, knobWidth, knobHeight);
        x += knobWidth + gap;
        loopDriveKnob.setBounds(x, bounds.getY() + 10, knobWidth, knobHeight);

        const int ySelectors = bounds.getY() + knobHeight + 30;
        loopSatBox.setBounds(bounds.getX() + 20, ySelectors, 100, 24);
        delayModeBox.setBounds(bounds.getX() + 140, ySelectors, 100, 24);
    }

private:
    PDLKnob feedbackKnob;
    PDLKnob dampKnob;
    PDLKnob loopCutKnob;
    PDLKnob crossFeedKnob;
    PDLKnob loopDriveKnob;
    ComboBox loopSatBox;
    ComboBox delayModeBox;
    MarsDSP::GUI::ConsoleButton delayModeButton;
    std::unique_ptr<AudioProcessorValueTreeState::ComboBoxAttachment> loopSatAttach;
    std::unique_ptr<AudioProcessorValueTreeState::ComboBoxAttachment> delayModeAttach;
};

// 3. DELAY -> MOD sub-panel
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
        auto bounds = getLocalBounds().reduced(16);
        constexpr int knobSize = 90;
        depthKnob.setBounds(bounds.getX() + 40, bounds.getY() + 20, knobSize, knobSize + 16);
        rateKnob.setBounds(bounds.getX() + 160, bounds.getY() + 20, knobSize, knobSize + 16);
    }

private:
    PDLKnob depthKnob;
    PDLKnob rateKnob;
};

// 4. CHARACTER -> DRIVE sub-panel
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
        auto bounds = getLocalBounds().reduced(16);
        constexpr int knobSize = 90;
        driveKnob.setBounds(bounds.getX() + 40, bounds.getY() + 20, knobSize, knobSize + 16);
        adaaOrderBox.setBounds(bounds.getX() + 160, bounds.getY() + 40, 100, 24);
    }

private:
    PDLKnob driveKnob;
    ComboBox adaaOrderBox;
    std::unique_ptr<AudioProcessorValueTreeState::ComboBoxAttachment> adaaAttach;
};

// 5. CHARACTER -> DIFFUSE sub-panel
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
        auto bounds = getLocalBounds().reduced(16);
        constexpr int knobSize = 85;
        constexpr int gap = 12;

        enableButton.setBounds(bounds.getX() + 10, bounds.getY() + 30, 28, 28);

        int x = bounds.getX() + 50;
        diffusionKnob.setBounds(x, bounds.getY() + 10, knobSize, knobSize + 14);
        x += knobSize + gap;
        sizeKnob.setBounds(x, bounds.getY() + 10, knobSize, knobSize + 14);
        x += knobSize + gap;
        modDepthKnob.setBounds(x, bounds.getY() + 10, knobSize, knobSize + 14);
        x += knobSize + gap;
        modRateKnob.setBounds(x, bounds.getY() + 10, knobSize, knobSize + 14);
    }

private:
    MarsDSP::GUI::PowerButton enableButton;
    PDLKnob diffusionKnob;
    PDLKnob sizeKnob;
    PDLKnob modDepthKnob;
    PDLKnob modRateKnob;
    std::unique_ptr<AudioProcessorValueTreeState::ButtonAttachment> enableAttach;
};

// 6. OUTPUT -> FILTER sub-panel
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
        auto bounds = getLocalBounds().reduced(16);
        constexpr int knobSize = 90;
        filterModeBox.setBounds(bounds.getX() + 20, bounds.getY() + 40, 100, 24);
        hpfKnob.setBounds(bounds.getX() + 140, bounds.getY() + 15, knobSize, knobSize + 16);
        lpfKnob.setBounds(bounds.getX() + 250, bounds.getY() + 15, knobSize, knobSize + 16);
    }

private:
    ComboBox filterModeBox;
    PDLKnob hpfKnob;
    PDLKnob lpfKnob;
    std::unique_ptr<AudioProcessorValueTreeState::ComboBoxAttachment> modeAttach;
};

// 7. OUTPUT -> LEVEL sub-panel
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

        bypassButton.setColours(GUIColours::accentRed, GUIColours::textDim);
        bypassAttach = std::make_unique<AudioProcessorValueTreeState::ButtonAttachment>(
            proc.getAPVTS(), bypassParamID.getParamID(), bypassButton);
        addAndMakeVisible(bypassButton);
    }

    void resized() override
    {
        auto bounds = getLocalBounds().reduced(16);
        constexpr int knobSize = 85;
        constexpr int gap = 14;

        int x = bounds.getX() + 20;
        mixKnob.setBounds(x, bounds.getY() + 10, knobSize, knobSize + 14);
        x += knobSize + gap;
        gainKnob.setBounds(x, bounds.getY() + 10, knobSize, knobSize + 14);
        x += knobSize + gap;
        bitsKnob.setBounds(x, bounds.getY() + 10, knobSize, knobSize + 14);

        bypassButton.setBounds(x + knobSize + 24, bounds.getY() + 35, 28, 28);
    }

private:
    PDLKnob mixKnob;
    PDLKnob gainKnob;
    PDLKnob bitsKnob;
    MarsDSP::GUI::PowerButton bypassButton;
    std::unique_ptr<AudioProcessorValueTreeState::ButtonAttachment> bypassAttach;
};

} // namespace

ChronosEditor::ChronosEditor(ChronosProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p)
{
    setLookAndFeel(&lnf_);

    pages_[0] = &delayPage_;
    pages_[1] = &characterPage_;
    pages_[2] = &outputPage_;

    delayPage_.addSubPanel("TAPS", std::make_unique<TapsPanel>(processorRef));
    delayPage_.addSubPanel("REPEATS", std::make_unique<RepeatsPanel>(processorRef));
    delayPage_.addSubPanel("MOD", std::make_unique<ModPanel>(processorRef));

    characterPage_.addSubPanel("DRIVE", std::make_unique<DrivePanel>(processorRef));
    characterPage_.addSubPanel("DIFFUSE", std::make_unique<DiffusePanel>(processorRef));

    outputPage_.addSubPanel("FILTER", std::make_unique<FilterPanel>(processorRef));
    outputPage_.addSubPanel("LEVEL", std::make_unique<LevelPanel>(processorRef));

    for (auto* page : pages_)
        addChildComponent(*page);

    const auto rawMode = processorRef.getParameters().getRawDelayMode();
    const auto delayDotCol = (rawMode > 0.5f) ? MarsDSP::GUI::Colours::accentDelayBBD
                                              : MarsDSP::GUI::Colours::accentDelayDigital;

    tabBar_.addTab("DELAY", delayDotCol);
    tabBar_.addTab("CHARACTER", MarsDSP::GUI::Colours::accentPurple);
    tabBar_.addTab("OUTPUT", MarsDSP::GUI::Colours::accentBlue);

    tabBar_.onTabChanged = [this](const int index)
    {
        setSelectedTab(index);
    };

    addAndMakeVisible(tabBar_);
    setSelectedTab(0);

    processorRef.getAPVTS().addParameterListener("delayMode", this);

    setResizable(true, true);
    setResizeLimits(600, 360, 1500, 900);
    getConstrainer()->setFixedAspectRatio(1000.0 / 600.0);
    setSize(1000, 600);
}

ChronosEditor::~ChronosEditor()
{
    processorRef.getAPVTS().removeParameterListener("delayMode", this);
    setLookAndFeel(nullptr);
}

void ChronosEditor::setSelectedTab(const int index)
{
    tabBar_.setSelectedTab(index);
    for (int i = 0; i < static_cast<int>(pages_.size()); ++i)
    {
        pages_[static_cast<std::size_t>(i)]->setVisible(i == index);
    }
    resized();
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
    const auto col = (delayModeVal > 0.5f) ? MarsDSP::GUI::Colours::accentDelayBBD
                                           : MarsDSP::GUI::Colours::accentDelayDigital;
    tabBar_.setTabDotColour(0, col);
}

void ChronosEditor::paint(Graphics& g)
{
    g.fillAll(MarsDSP::GUI::Colours::background);

    const auto headerBounds = getLocalBounds().withHeight(44).toFloat();
    g.setColour(MarsDSP::GUI::Colours::headerBackground);
    g.fillRect(headerBounds);

    g.setColour(MarsDSP::GUI::Colours::panelBorder);
    g.drawHorizontalLine(44, 0.0f, static_cast<float>(getWidth()));
}

void ChronosEditor::resized()
{
    constexpr int headerHeight = 44;
    constexpr int tabBarWidth = 360;
    constexpr int tabBarHeight = 28;

    tabBar_.setBounds(16, (headerHeight - tabBarHeight) / 2, tabBarWidth, tabBarHeight);

    const auto contentBounds = getLocalBounds().withTrimmedTop(headerHeight + 8).reduced(12, 8);
    for (auto* page : pages_)
    {
        if (page->isVisible())
            page->setBounds(contentBounds);
    }
}
