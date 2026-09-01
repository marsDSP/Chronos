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
    setColour(TooltipWindow::backgroundColourId, Colours::panelBackground);
    setColour(TooltipWindow::textColourId, Colours::textBright);
    setColour(TooltipWindow::outlineColourId, Colours::panelBorder);
}

void LookAndFeel::setMetrics(const Metrics& m)
{
    metrics_ = m;
}

Font LookAndFeel::getLabelFont(Label&)
{
    return Fonts::font(Fonts::Weight::Regular, metrics_.font(13.0f));
}

Font LookAndFeel::getComboBoxFont(ComboBox&)
{
    return Fonts::font(Fonts::Weight::Regular, metrics_.font(12.0f));
}

Font LookAndFeel::getPopupMenuFont()
{
    return Fonts::font(Fonts::Weight::Regular, metrics_.font(14.0f));
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

    const float arcThickness = metrics_.stroke(Metrics::kGroupStroke);
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
    const float outline = metrics_.stroke(Metrics::kIconStroke);
    g.drawEllipse(centreX - knobRadius, centreY - knobRadius, knobDiameter, knobDiameter, outline);

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

    const auto m = metrics_;
    const auto bounds = Rectangle<int>(0, 0, width, height).toFloat();
    const float corner = m.pxf(Metrics::kCornerSmall);
    const float sw = m.stroke(Metrics::kHairline);

    g.setColour(Colours::dropdownBg);
    g.fillRoundedRectangle(bounds, corner);
    g.setColour(Colours::dropdownBorder);
    g.drawRoundedRectangle(bounds.reduced(sw / 2), corner, sw);

    const auto arrowX = static_cast<float>(width) - m.pxf(Metrics::kComboArrowInset);
    const auto arrowY = static_cast<float>(height) * 0.5f;

    Path arrow;
    arrow.addTriangle(arrowX - 4.0f, arrowY - 2.0f,
                      arrowX + 4.0f, arrowY - 2.0f,
                      arrowX, arrowY + 3.0f);

    g.setColour(Colours::textDim);
    g.fillPath(arrow);
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
    const auto m = metrics_;
    const float corner = m.pxf(Metrics::kCornerSmall);
    const float sw = m.stroke(Metrics::kHairline);
    const float half = sw / 2;

    g.setColour(Colours::panelBackground);
    g.fillRoundedRectangle(Rectangle<float>(static_cast<float>(width), static_cast<float>(height)), corner);

    g.setColour(Colours::panelBorder);
    g.drawRoundedRectangle(half, half, static_cast<float>(width) - sw, static_cast<float>(height) - sw, corner, sw);
}

void LookAndFeel::drawTooltip(Graphics& g, const String& text, int width, int height)
{
    const Font f = Fonts::font(Fonts::Weight::Regular, metrics_.font(11.0f));

    g.fillAll(findColour(TooltipWindow::backgroundColourId));

   #if ! JUCE_MAC
    g.setColour(findColour(TooltipWindow::outlineColourId));
    g.drawRect(0, 0, width, height, 1);
   #endif

    g.setColour(findColour(TooltipWindow::textColourId));
    g.setFont(f);
    g.drawText(text, 4, 0, width - 8, height, Justification::centredLeft, true);
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

    const auto m = metrics_;

    if (isSeparator)
    {
        auto r = area.reduced(m.px(Metrics::kMenuSeparatorInset), 0);
        r.removeFromTop(roundToInt(static_cast<float>(r.getHeight()) * 0.5f) - 1);
        g.setColour(Colours::panelBorder);
        g.fillRect(r.removeFromTop(1));
        return;
    }

    auto r = area.reduced(2);
    const float highlightCorner = m.pxf(3.0f);
    if (isHighlighted && isActive)
    {
        g.setColour(Colours::headerBackground);
        g.fillRoundedRectangle(r.toFloat(), highlightCorner);
        g.setColour(Colours::textBright);
    }
    else
    {
        g.setColour(isActive ? Colours::textBright : Colours::textDim);
    }

    r.removeFromLeft(m.px(Metrics::kMenuItemInset));
    g.setFont(getPopupMenuFont());
    g.drawFittedText(text, r, Justification::centredLeft, 1);

    if (hasSubMenu)
    {
        Path arrow;
        const auto arrowArea = r.removeFromRight(m.px(Metrics::kMenuArrowBox)).withSizeKeepingCentre(4, 7).toFloat();
        arrow.addTriangle(arrowArea.getTopLeft(),
                          arrowArea.getBottomLeft(),
                          Point<float>(arrowArea.getRight(), arrowArea.getCentreY()));

        g.setColour(isHighlighted ? Colours::textBright : Colours::textPrimary);
        g.fillPath(arrow);
    }
}

} // namespace MarsDSP::GUI
