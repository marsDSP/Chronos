#include "ChronosEditor.h"
#include "PluginParameters.h"
#include "gui/ChronosColours.h"

using namespace juce;

namespace
{
    // Editor geometry. Picked so every control fits without resizing.
    constexpr int kWindowW   = 1000;
    constexpr int kWindowH   = 560;
    constexpr int kHeaderH   = 48;
    constexpr int kPad       = 8;
    constexpr int kGroupTop  = 22; // room for the GroupComponent's header text
    constexpr int kGroupSide = 12;
    constexpr int kGroupBot  = 10;

    // Lay out a sequence of components into a `cols x rows` grid that fills
    // `area`. Components map row-major from top-left. Null entries leave
    // their cell empty (used to keep button cells aligned with knob cells).
    void gridLayout(Rectangle<int> area,
                    std::initializer_list<Component*> items,
                    int cols, int rows)
    {
        const int n     = static_cast<int>(items.size());
        const int cellW = area.getWidth()  / cols;
        const int cellH = area.getHeight() / rows;

        int i = 0;
        for (auto* c : items)
        {
            const int col = i % cols;
            const int row = i / cols;
            if (c != nullptr)
                c->setBounds(area.getX() + col * cellW,
                             area.getY() + row * cellH,
                             cellW, cellH);
            ++i;
            if (i >= n) break;
        }
    }

    // Reduce a panel's outer rect down to its usable knob area.
    Rectangle<int> innerOf(Rectangle<int> r)
    {
        return r.withTrimmedTop(kGroupTop)
                .withTrimmedBottom(kGroupBot)
                .withTrimmedLeft(kGroupSide)
                .withTrimmedRight(kGroupSide);
    }
} // namespace

//==============================================================================
ChronosEditor::ChronosEditor(ChronosProcessor& p)
    : AudioProcessorEditor(&p), proc(p)
    // ----- Delay (green) -----
    , delayTimeKnob ("DELAY",     p.apvts, ParameterID(Chronos::ParamID::kDelayTime, 1), Chronos::ChronosColours::accentGreen)
    , mixKnob       ("MIX",       p.apvts, ParameterID(Chronos::ParamID::kMix, 1),       Chronos::ChronosColours::accentGreen)
    , feedbackKnob  ("FEEDBACK",  p.apvts, ParameterID(Chronos::ParamID::kFeedback, 1),  Chronos::ChronosColours::accentGreen)
    , lowCutKnob    ("LOW CUT",   p.apvts, ParameterID(Chronos::ParamID::kLowCut, 1),    Chronos::ChronosColours::accentGreen)
    , highCutKnob   ("HIGH CUT",  p.apvts, ParameterID(Chronos::ParamID::kHighCut, 1),   Chronos::ChronosColours::accentGreen)
    , crossfeedKnob ("CROSSFEED", p.apvts, ParameterID(Chronos::ParamID::kCrossfeed, 1), Chronos::ChronosColours::accentGreen)
    // ----- Reverb (blue) -----
    , revMixKnob      ("MIX",       p.apvts, ParameterID(Chronos::ParamID::kReverbMix, 1),                  Chronos::ChronosColours::accentBlue)
    , revRoomKnob     ("SIZE",      p.apvts, ParameterID(Chronos::ParamID::kReverbRoomSize, 1),             Chronos::ChronosColours::accentBlue)
    , revDecayKnob    ("DECAY",     p.apvts, ParameterID(Chronos::ParamID::kReverbDecayTime, 1),            Chronos::ChronosColours::accentBlue)
    , revPredelayKnob ("PREDELAY",  p.apvts, ParameterID(Chronos::ParamID::kReverbPredelay, 1),             Chronos::ChronosColours::accentBlue)
    , revDiffKnob     ("DIFFUSION", p.apvts, ParameterID(Chronos::ParamID::kReverbDiffusion, 1),            Chronos::ChronosColours::accentBlue)
    , revBuildKnob    ("BUILDUP",   p.apvts, ParameterID(Chronos::ParamID::kReverbBuildup, 1),              Chronos::ChronosColours::accentBlue)
    , revModKnob      ("MOD",       p.apvts, ParameterID(Chronos::ParamID::kReverbModulation, 1),           Chronos::ChronosColours::accentBlue)
    , revHfDampKnob   ("HF DAMP",   p.apvts, ParameterID(Chronos::ParamID::kReverbHighFrequencyDamping, 1), Chronos::ChronosColours::accentBlue)
    , revLfDampKnob   ("LF DAMP",   p.apvts, ParameterID(Chronos::ParamID::kReverbLowFrequencyDamping, 1),  Chronos::ChronosColours::accentBlue)
    // ----- Wow / Flutter (purple) -----
    , wowRateKnob      ("WOW RATE",  p.apvts, ParameterID(Chronos::ParamID::kWowRate, 1),      Chronos::ChronosColours::accentPurple)
    , wowDepthKnob     ("WOW DEPTH", p.apvts, ParameterID(Chronos::ParamID::kWowDepth, 1),     Chronos::ChronosColours::accentPurple)
    , wowDriftKnob     ("WOW DRIFT", p.apvts, ParameterID(Chronos::ParamID::kWowDrift, 1),     Chronos::ChronosColours::accentPurple)
    , flutterRateKnob  ("FL RATE",   p.apvts, ParameterID(Chronos::ParamID::kFlutterRate, 1),  Chronos::ChronosColours::accentPurple)
    , flutterDepthKnob ("FL DEPTH",  p.apvts, ParameterID(Chronos::ParamID::kFlutterDepth, 1), Chronos::ChronosColours::accentPurple)
    // ----- Ducker (orange) -----
    , duckThreshKnob  ("THRESH",  p.apvts, ParameterID(Chronos::ParamID::kDuckerThreshold, 1), Chronos::ChronosColours::accentOrange)
    , duckAmountKnob  ("AMOUNT",  p.apvts, ParameterID(Chronos::ParamID::kDuckerAmount, 1),    Chronos::ChronosColours::accentOrange)
    , duckAttackKnob  ("ATTACK",  p.apvts, ParameterID(Chronos::ParamID::kDuckerAttack, 1),    Chronos::ChronosColours::accentOrange)
    , duckReleaseKnob ("RELEASE", p.apvts, ParameterID(Chronos::ParamID::kDuckerRelease, 1),   Chronos::ChronosColours::accentOrange)
{
    using namespace Chronos;

    // ---- Title ----
    titleLabel.setText("CHRONOS", dontSendNotification);
    titleLabel.setFont(Font(FontOptions(22.0f, Font::bold)));
    titleLabel.setColour(Label::textColourId, ChronosColours::textBright);
    titleLabel.setJustificationType(Justification::centredLeft);
    addAndMakeVisible(titleLabel);

    syncIntervalLabel.setText("DIV", dontSendNotification);
    syncIntervalLabel.setFont(Font(FontOptions(11.0f, Font::bold)));
    syncIntervalLabel.setColour(Label::textColourId, ChronosColours::textDim);
    syncIntervalLabel.setJustificationType(Justification::centredRight);
    addAndMakeVisible(syncIntervalLabel);

    // ---- Section frames ----
    auto styleGroup = [](GroupComponent& g, Colour accent)
    {
        g.setColour(GroupComponent::outlineColourId, ChronosColours::panelBorder);
        g.setColour(GroupComponent::textColourId,    accent);
        g.setTextLabelPosition(Justification::centredTop);
    };
    styleGroup(delayGroup,  ChronosColours::accentGreen);
    styleGroup(reverbGroup, ChronosColours::accentBlue);
    styleGroup(modGroup,    ChronosColours::accentPurple);
    styleGroup(duckerGroup, ChronosColours::accentOrange);

    addAndMakeVisible(delayGroup);
    addAndMakeVisible(reverbGroup);
    addAndMakeVisible(modGroup);
    addAndMakeVisible(duckerGroup);

    // ---- Knobs visible ----
    for (auto* k : { &delayTimeKnob, &mixKnob, &feedbackKnob, &lowCutKnob, &highCutKnob, &crossfeedKnob })
        addAndMakeVisible(k);
    for (auto* k : { &revMixKnob, &revRoomKnob, &revDecayKnob, &revPredelayKnob, &revDiffKnob,
                     &revBuildKnob, &revModKnob, &revHfDampKnob, &revLfDampKnob })
        addAndMakeVisible(k);
    for (auto* k : { &wowRateKnob, &wowDepthKnob, &wowDriftKnob, &flutterRateKnob, &flutterDepthKnob })
        addAndMakeVisible(k);
    for (auto* k : { &duckThreshKnob, &duckAmountKnob, &duckAttackKnob, &duckReleaseKnob })
        addAndMakeVisible(k);

    // ---- Toggles ----
    auto styleToggle = [](ToggleButton& b, Colour accent)
    {
        b.setColour(ToggleButton::textColourId,         ChronosColours::textBright);
        b.setColour(ToggleButton::tickColourId,         accent);
        b.setColour(ToggleButton::tickDisabledColourId, ChronosColours::textDim);
    };
    styleToggle(bypassButton,        ChronosColours::accentRed);
    styleToggle(monoButton,          ChronosColours::accentYellow);
    styleToggle(syncButton,          ChronosColours::accentYellow);
    styleToggle(revBypassButton,     ChronosColours::accentBlue);
    styleToggle(flutterEnableButton, ChronosColours::accentPurple);
    styleToggle(duckerBypassButton,  ChronosColours::accentOrange);

    addAndMakeVisible(bypassButton);
    addAndMakeVisible(monoButton);
    addAndMakeVisible(syncButton);
    addAndMakeVisible(revBypassButton);
    addAndMakeVisible(flutterEnableButton);
    addAndMakeVisible(duckerBypassButton);

    // ---- Sync interval combo ----
    syncIntervalCombo.setColour(ComboBox::backgroundColourId, ChronosColours::dropdownBg);
    syncIntervalCombo.setColour(ComboBox::outlineColourId,    ChronosColours::dropdownBorder);
    syncIntervalCombo.setColour(ComboBox::textColourId,       ChronosColours::textBright);
    syncIntervalCombo.setColour(ComboBox::arrowColourId,      ChronosColours::textPrimary);

    if (auto* choiceParam = dynamic_cast<AudioParameterChoice*>(p.apvts.getParameter(ParamID::kSyncInterval)))
        syncIntervalCombo.addItemList(choiceParam->choices, 1);
    addAndMakeVisible(syncIntervalCombo);

    // ---- APVTS attachments ----
    bypassAtt       = std::make_unique<BtnAtt>(p.apvts, ParamID::kBypass,        bypassButton);
    monoAtt         = std::make_unique<BtnAtt>(p.apvts, ParamID::kMono,          monoButton);
    syncAtt         = std::make_unique<BtnAtt>(p.apvts, ParamID::kSyncEnabled,   syncButton);
    revBypassAtt    = std::make_unique<BtnAtt>(p.apvts, ParamID::kReverbBypass,  revBypassButton);
    flutterAtt      = std::make_unique<BtnAtt>(p.apvts, ParamID::kFlutterOnOff,  flutterEnableButton);
    duckerBypassAtt = std::make_unique<BtnAtt>(p.apvts, ParamID::kDuckerBypass,  duckerBypassButton);
    syncIntervalAtt = std::make_unique<BoxAtt>(p.apvts, ParamID::kSyncInterval,  syncIntervalCombo);

    setSize(kWindowW, kWindowH);
}

ChronosEditor::~ChronosEditor() = default;

//==============================================================================
void ChronosEditor::paint(Graphics& g)
{
    using namespace Chronos;

    g.fillAll(ChronosColours::background);

    // Header strip
    auto header = getLocalBounds().removeFromTop(kHeaderH);
    g.setColour(ChronosColours::headerBackground);
    g.fillRect(header);
    g.setColour(ChronosColours::panelBorder);
    g.fillRect(header.removeFromBottom(1));
}

void ChronosEditor::resized()
{
    auto bounds = getLocalBounds();

    // ---- Header ----
    auto header = bounds.removeFromTop(kHeaderH).reduced(kPad, 6);
    titleLabel.setBounds(header.removeFromLeft(160));

    // Master controls, right-aligned. removeFromRight pulls in reverse order.
    bypassButton.setBounds     (header.removeFromRight(80));
    monoButton.setBounds       (header.removeFromRight(70));
    header.removeFromRight(8); // gap
    syncIntervalCombo.setBounds(header.removeFromRight(120).reduced(2, 4));
    syncIntervalLabel.setBounds(header.removeFromRight(32));
    syncButton.setBounds       (header.removeFromRight(70));

    // ---- Body: 2 x 2 of section panels ----
    bounds.reduce(kPad, kPad);
    const int colW = (bounds.getWidth()  - kPad) / 2;
    const int rowH = (bounds.getHeight() - kPad) / 2;

    auto delayArea  = Rectangle<int>(bounds.getX(),               bounds.getY(),               colW, rowH);
    auto reverbArea = Rectangle<int>(bounds.getX() + colW + kPad, bounds.getY(),               colW, rowH);
    auto modArea    = Rectangle<int>(bounds.getX(),               bounds.getY() + rowH + kPad, colW, rowH);
    auto duckerArea = Rectangle<int>(bounds.getX() + colW + kPad, bounds.getY() + rowH + kPad, colW, rowH);

    delayGroup .setBounds(delayArea);
    reverbGroup.setBounds(reverbArea);
    modGroup   .setBounds(modArea);
    duckerGroup.setBounds(duckerArea);

    // ---- Lay out controls inside each panel ----
    // Delay: 6 knobs in a 3 x 2 grid.
    gridLayout(innerOf(delayArea),
               { &delayTimeKnob, &mixKnob,    &feedbackKnob,
                 &lowCutKnob,    &highCutKnob, &crossfeedKnob },
               3, 2);

    // Reverb: 9 knobs + bypass in a 5 x 2 grid (last cell is the toggle).
    gridLayout(innerOf(reverbArea),
               { &revMixKnob,   &revRoomKnob, &revDecayKnob,  &revPredelayKnob, &revDiffKnob,
                 &revBuildKnob, &revModKnob,  &revHfDampKnob, &revLfDampKnob,   &revBypassButton },
               5, 2);

    // Wow & Flutter: 5 knobs + flutter toggle in a 3 x 2 grid.
    gridLayout(innerOf(modArea),
               { &wowRateKnob,     &wowDepthKnob,    &wowDriftKnob,
                 &flutterRateKnob, &flutterDepthKnob, &flutterEnableButton },
               3, 2);

    // Ducker: 4 knobs + bypass toggle. Use 3 x 2 with one empty cell so the
    // knobs stay the same size as the modulation panel above for visual rhyme.
    gridLayout(innerOf(duckerArea),
               { &duckThreshKnob, &duckAmountKnob,     &duckAttackKnob,
                 &duckReleaseKnob, &duckerBypassButton, nullptr },
               3, 2);
}
