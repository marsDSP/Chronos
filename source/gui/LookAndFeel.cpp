#include "LookAndFeel.h"
#include "Fonts.h"
#include "Metrics.h"

namespace MarsDSP::GUI {

LookAndFeel::LookAndFeel()
{
    setColour(ResizableWindow::backgroundColourId, Colours::background);
    setColour(Label::textColourId, Colours::textPrimary);
    setColour(Slider::rotarySliderFillColourId, Colours::accentDelayDigital);
    setColour(Slider::rotarySliderOutlineColourId, Colours::knobTrack);
    setColour(Slider::thumbColourId, Colours::knobGradientTop);
    setColour(ComboBox::backgroundColourId, Colours::dropdownBg);
    setColour(ComboBox::outlineColourId, Colours::dropdownBorder);
    setColour(ComboBox::textColourId, Colours::textPrimary);
    setColour(ComboBox::arrowColourId, Colours::textDim);
    setColour(PopupMenu::backgroundColourId, Colours::panelBackground);
    setColour(PopupMenu::textColourId, Colours::textPrimary);
    setColour(PopupMenu::highlightedBackgroundColourId, Colours::headerBackground);
    setColour(PopupMenu::highlightedTextColourId, Colours::textBright);
}

Font LookAndFeel::getLabelFont(Label&)
{
    return Fonts::font(Fonts::Weight::Regular, currentMetrics().font(13.0f));
}

Font LookAndFeel::getComboBoxFont(ComboBox&)
{
    return Fonts::font(Fonts::Weight::Regular, currentMetrics().font(12.0f));
}

Font LookAndFeel::getPopupMenuFont()
{
    return Fonts::font(Fonts::Weight::Regular, currentMetrics().font(14.0f));
}

void LookAndFeel::drawRotarySlider(Graphics& g,
                                  const int x,
                                  const int y,
                                  const int width,
                                  const int height,
                                  const float sliderPos,
                                  const float rotaryStartAngle,
                                  const float rotaryEndAngle,
                                  Slider& slider)
{
    const auto bounds = Rectangle<int>(x, y, width, height).toFloat().reduced(4.0f);
    const auto radius = jmin(bounds.getWidth(), bounds.getHeight()) * 0.5f;
    const auto centreX = bounds.getCentreX();
    const auto centreY = bounds.getCentreY();
    const auto angle = rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);

    constexpr float arcThickness = 2.0f;
    Path trackArc;
    trackArc.addCentredArc(centreX, centreY, radius, radius, 0.0f, rotaryStartAngle, rotaryEndAngle, true);

    g.setColour(Colours::knobTrack);
    g.strokePath(trackArc, PathStrokeType(arcThickness, PathStrokeType::curved, PathStrokeType::rounded));

    if (sliderPos > 0.0f)
    {
        Path valueArc;
        valueArc.addCentredArc(centreX, centreY, radius, radius, 0.0f, rotaryStartAngle, angle, true);
        g.setColour(slider.findColour(Slider::rotarySliderFillColourId));
        g.strokePath(valueArc, PathStrokeType(arcThickness, PathStrokeType::curved, PathStrokeType::rounded));
    }

    const auto knobRadius = radius * 0.62f;
    const auto knobDiameter = knobRadius * 2.0f;

    const ColourGradient knobGrad(Colours::knobGradientTop,
                                  centreX,
                                  centreY - knobRadius,
                                  Colours::knobGradientBot,
                                  centreX,
                                  centreY + knobRadius,
                                  false);

    g.setGradientFill(knobGrad);
    g.fillEllipse(centreX - knobRadius, centreY - knobRadius, knobDiameter, knobDiameter);

    g.setColour(Colour(0x08FFFFFF));
    g.drawEllipse(centreX - knobRadius, centreY - knobRadius, knobDiameter, knobDiameter, 1.5f);

    Path pointer;
    const auto pointerLength = knobRadius * 0.6f;
    constexpr auto pointerThickness = 2.0f;
    pointer.addRoundedRectangle(-pointerThickness * 0.5f, -knobRadius + 2.0f, pointerThickness, pointerLength, 1.0f);
    pointer.applyTransform(AffineTransform::rotation(angle).translated(centreX, centreY));

    g.setColour(Colours::textPrimary);
    g.fillPath(pointer);
}

void LookAndFeel::drawComboBox(Graphics& g,
                              const int width,
                              const int height,
                              bool isButtonDown,
                              int buttonX,
                              int buttonY,
                              int buttonW,
                              int buttonH,
                              ComboBox& box)
{
    ignoreUnused(isButtonDown, buttonX, buttonY, buttonW, buttonH, box);

    const auto bounds = Rectangle<int>(0, 0, width, height).toFloat();

    g.setColour(Colours::dropdownBg);
    g.fillRoundedRectangle(bounds, 4.0f);
    g.setColour(Colours::dropdownBorder);
    g.drawRoundedRectangle(bounds.reduced(0.5f), 4.0f, 1.0f);

    const auto arrowX = static_cast<float>(width) - 20.0f;
    const auto arrowY = static_cast<float>(height) * 0.5f;

    Path arrow;
    arrow.addTriangle(arrowX - 4.0f, arrowY - 2.0f,
                      arrowX + 4.0f, arrowY - 2.0f,
                      arrowX, arrowY + 3.0f);

    g.setColour(Colours::textDim);
    g.fillPath(arrow);
}

void LookAndFeel::drawGroupComponentOutline(Graphics& g,
                                            const int width,
                                            const int height,
                                            const String& text,
                                            const Justification& position,
                                            GroupComponent& group)
{
    constexpr float textH = 15.0f;
    constexpr float indent = 3.0f;
    constexpr float textEdgeGap = 4.0f;
    auto cornerSize = 5.0f;

    const Font font = Fonts::font(Fonts::Weight::Semibold, currentMetrics().font(textH));
    constexpr auto x = indent;
    const auto y = font.getAscent() - 3.0f;
    const auto w = std::max(0.0f, static_cast<float>(width) - x * 2.0f);
    const auto h = std::max(0.0f, static_cast<float>(height) - y - indent);

    cornerSize = jmin(cornerSize, w * 0.5f, h * 0.5f);
    const auto cs2 = 2.0f * cornerSize;

    const auto textW = text.isEmpty() ? 0.0f
                                      : std::clamp(font.getStringWidthFloat(text) + textEdgeGap * 2.0f,
                                                   0.0f,
                                                   std::max(0.0f, w - cs2 - textEdgeGap * 2.0f));

    auto textX = cornerSize + textEdgeGap;
    if (position.testFlags(Justification::horizontallyCentred))
        textX = cornerSize + (w - cs2 - textW) * 0.5f;
    else if (position.testFlags(Justification::right))
        textX = w - cornerSize - textW - textEdgeGap;

    Path p;
    p.startNewSubPath(x + textX + textW, y);
    p.lineTo(x + w - cornerSize, y);
    p.addArc(x + w - cs2, y, cs2, cs2, 0.0f, MathConstants<float>::pi * 0.5f);
    p.lineTo(x + w, y + h - cornerSize);
    p.addArc(x + w - cs2, y + h - cs2, cs2, cs2, MathConstants<float>::pi * 0.5f, MathConstants<float>::pi);
    p.lineTo(x + cornerSize, y + h);
    p.addArc(x, y + h - cs2, cs2, cs2, MathConstants<float>::pi, MathConstants<float>::pi * 1.5f);
    p.lineTo(x, y + cornerSize);
    p.addArc(x, y, cs2, cs2, MathConstants<float>::pi * 1.5f, MathConstants<float>::twoPi);
    p.lineTo(x + textX, y);

    const auto alpha = group.isEnabled() ? 1.0f : 0.5f;
    g.setColour(group.findColour(GroupComponent::outlineColourId).withMultipliedAlpha(alpha));
    g.strokePath(p, PathStrokeType(2.0f));

    g.setColour(group.findColour(GroupComponent::textColourId).withMultipliedAlpha(alpha));
    g.setFont(font);
    g.drawText(text,
               roundToInt(x + textX), 0,
               roundToInt(textW),
               roundToInt(textH),
               Justification::centred, true);
}

void LookAndFeel::drawLabel(Graphics& g, Label& label)
{
    g.fillAll(label.findColour(Label::backgroundColourId));

    if (!label.isBeingEdited())
    {
        const auto alpha = label.isEnabled() ? 1.0f : 0.5f;
        const auto font = getLabelFont(label);

        g.setColour(label.findColour(Label::textColourId).withMultipliedAlpha(alpha));
        g.setFont(font);

        const auto textArea = getLabelBorderSize(label).subtractedFrom(label.getLocalBounds());
        g.drawFittedText(label.getText(),
                         textArea,
                         label.getJustificationType(),
                         jmax(1, static_cast<int>(static_cast<float>(textArea.getHeight()) / font.getHeight())),
                         label.getMinimumHorizontalScale());
    }
}

void LookAndFeel::drawPopupMenuBackground(Graphics& g, const int width, const int height)
{
    g.setColour(Colours::panelBackground);
    g.fillRoundedRectangle(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 4.0f);

    g.setColour(Colours::panelBorder);
    g.drawRoundedRectangle(0.5f, 0.5f, static_cast<float>(width) - 1.0f, static_cast<float>(height) - 1.0f, 4.0f, 1.0f);
}

void LookAndFeel::drawPopupMenuItem(Graphics& g,
                                    const Rectangle<int>& area,
                                    const bool isSeparator,
                                    const bool isActive,
                                    const bool isHighlighted,
                                    const bool isTicked,
                                    const bool hasSubMenu,
                                    const String& text,
                                    const String& shortcutKeyText,
                                    const Drawable* icon,
                                    const Colour* textColour)
{
    ignoreUnused(shortcutKeyText, icon, textColour, isTicked);

    if (isSeparator)
    {
        auto r = area.reduced(14, 0);
        r.removeFromTop(roundToInt(static_cast<float>(r.getHeight()) * 0.5f) - 1);
        g.setColour(Colours::panelBorder);
        g.fillRect(r.removeFromTop(1));
        return;
    }

    auto r = area.reduced(2);
    if (isHighlighted && isActive)
    {
        g.setColour(Colours::headerBackground);
        g.fillRoundedRectangle(r.toFloat(), 3.0f);
        g.setColour(Colours::textBright);
    }
    else
    {
        g.setColour(isActive ? Colours::textBright : Colours::textDim);
    }

    r.removeFromLeft(12);
    g.setFont(getPopupMenuFont());
    g.drawFittedText(text, r, Justification::centredLeft, 1);

    if (hasSubMenu)
    {
        Path arrow;
        const auto arrowArea = r.removeFromRight(16).withSizeKeepingCentre(4, 7).toFloat();
        arrow.addTriangle(arrowArea.getTopLeft(),
                          arrowArea.getBottomLeft(),
                          Point<float>(arrowArea.getRight(), arrowArea.getCentreY()));

        g.setColour(isHighlighted ? Colours::textBright : Colours::textPrimary);
        g.fillPath(arrow);
    }
}

} // namespace MarsDSP::GUI
