#include "ChronosEditor.h"

ChronosEditor::ChronosEditor(ChronosProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p)
{
    setLookAndFeel(&lnf_);

    pages_[0] = &delayPage_;
    pages_[1] = &characterPage_;
    pages_[2] = &outputPage_;

    delayPage_.addSubPanel("TAPS", nullptr);
    delayPage_.addSubPanel("REPEATS", nullptr);
    delayPage_.addSubPanel("MOD", nullptr);

    characterPage_.addSubPanel("DRIVE", nullptr);
    characterPage_.addSubPanel("DIFFUSE", nullptr);

    outputPage_.addSubPanel("FILTER", nullptr);
    outputPage_.addSubPanel("LEVEL", nullptr);

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
