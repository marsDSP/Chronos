#pragma once

#ifndef CHRONOS_PEDALKNOB_H
#define CHRONOS_PEDALKNOB_H

#include <JuceHeader.h>
#include "Colours.h"

namespace MarsDSP::GUI::Knobs {
    class PedalKnob : public LookAndFeel_V4 {
    public:
        PedalKnob()
        {
            ellipse2 = ImageCache::getFromMemory(BinaryData::Ellipse_2_png, BinaryData::Ellipse_2_pngSize);
            ellipse3 = ImageCache::getFromMemory(BinaryData::Ellipse_3_png, BinaryData::Ellipse_3_pngSize);
            ellipse4 = ImageCache::getFromMemory(BinaryData::Ellipse_4_png, BinaryData::Ellipse_4_pngSize);
            rectangle2 = ImageCache::getFromMemory(BinaryData::Rectangle_2_png, BinaryData::Rectangle_2_pngSize);
        }

        void drawRotarySlider(Graphics &g,
                              const int x,
                              const int y,
                              const int width,
                              const int height,
                              const float sliderPos,
                              const float rotaryStartAngle,
                              const float rotaryEndAngle,
                              Slider&) override
        {
            const auto radius = std::min((height / 2.0f), (width / 2.0f));
            const auto centreX = x + width * 0.5f;
            const auto centreY = y + height * 0.5f;
            const auto rw = radius * 2.0f;
            const auto angle = rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);

            auto drawImage = [&](const Image &img,
                                 const float cx,
                                 const float cy,
                                 const float scaleFactor,
                                 const float rotationAngle)
            {
                if (img.isValid())
                {
                    AffineTransform t;
                    t = t.translated(-cx, -cy);
                    if (rotationAngle != 0.0f) t = t.rotated(rotationAngle);
                    t = t.scaled(scaleFactor);
                    t = t.translated(centreX, centreY);
                    g.drawImageTransformed(img, t);
                }
            };

            // main knob canvas is 431 px wide in the source artwork
            const float baseScale = rw / 431.0f;

            drawImage(ellipse2, 215.0f, 157.5f, baseScale, 0.0f);
            drawImage(ellipse3, ellipse3.getWidth() * 0.5f, ellipse3.getHeight() * 0.5f, baseScale, 0.0f);
            drawImage(ellipse4, ellipse4.getWidth() * 0.5f, ellipse4.getHeight() * 0.5f, baseScale, angle);

            if (rectangle2.isValid())
            {
                AffineTransform t;
                t = t.translated(-rectangle2.getWidth() * 0.5f, -rectangle2.getHeight() * 0.5f);
                t = t.translated(0.0f, -80.0f);
                t = t.rotated(angle);
                t = t.scaled(baseScale);
                t = t.translated(centreX, centreY);
                g.drawImageTransformed(rectangle2, t);
            }
        }

    private:
        Image ellipse2;
        Image ellipse3;
        Image ellipse4;
        Image rectangle2;
    };

    namespace KnobHelpers
    {
        inline std::pair<float, float>getArcAngles(const float sliderPos,
                                                    const float rotaryStartAngle,
                                                    const float rotaryEndAngle,
                                                    const Slider &slider)
        {
            const auto angle = rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);

            if (slider.getProperties().getWithDefault("drawFromMiddle", false))
            {
                constexpr float middlePos = 0.5f;
                const auto middleAngle = rotaryStartAngle + middlePos * (rotaryEndAngle - rotaryStartAngle);
                return {std::min(angle, middleAngle), std::max(angle, middleAngle)};
            }

            return {rotaryStartAngle, angle};
        }
    }

    class PDLKnob : public Component
    {
    public:
        PDLKnob(const String &labelText,
                AudioProcessorValueTreeState &state,
                const ParameterID &pid,
                Colour arc = Colours::accentGreen);

        ~PDLKnob() override;

        void paint(Graphics &g) override;
        void paintOverChildren(Graphics &g) override;
        void resized() override;
        Slider &getSlider() { return slider; }
        void setLabelText(const String &text) { label.setText(text, dontSendNotification); }
        void setArcColour(Colour c);
        void setDrawFromMiddle(bool v) { slider.getProperties().set("drawFromMiddle", v); }

    private:
        Slider slider;
        Label label;
        Label valueLabel;
        Colour arcColour;
        PedalKnob lnf;
        std::unique_ptr<AudioProcessorValueTreeState::SliderAttachment> attachment;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PDLKnob)
    };
}
#endif
