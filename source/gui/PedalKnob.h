#pragma once

#ifndef CHRONOS_PEDALKNOB_H
#define CHRONOS_PEDALKNOB_H

#include <JuceHeader.h>
#include "Colours.h"
#include "Fonts.h"
#include "Metrics.h"

namespace MarsDSP::GUI::Knobs {
    // Resample a source image to a target pixel size with high quality.
    inline Image resampleTo(const Image& src, const int targetW, const int targetH)
    {
        if (! src.isValid() || targetW <= 0 || targetH <= 0)
            return {};
        Image dst(Image::ARGB, targetW, targetH, true);
        Graphics g(dst);
        g.setImageResamplingQuality(Graphics::highResamplingQuality);
        g.drawImage(src, 0, 0, targetW, targetH, 0, 0, src.getWidth(), src.getHeight());
        return dst;
    }

    class PedalKnob : public LookAndFeel_V4 {
    public:
        PedalKnob()
        {
            srcEllipse2 = ImageCache::getFromMemory(BinaryData::Ellipse_2_png, BinaryData::Ellipse_2_pngSize);
            srcEllipse3 = ImageCache::getFromMemory(BinaryData::Ellipse_3_png, BinaryData::Ellipse_3_pngSize);
            srcEllipse4 = ImageCache::getFromMemory(BinaryData::Ellipse_4_png, BinaryData::Ellipse_4_pngSize);
            srcRectangle2 = ImageCache::getFromMemory(BinaryData::Rectangle_2_png, BinaryData::Rectangle_2_pngSize);
        }

        void drawRotarySlider(Graphics &g,
                              const int x,
                              const int y,
                              const int width,
                              const int height,
                              const float sliderPos,
                              const float rotaryStartAngle,
                              const float rotaryEndAngle,
                              Slider& slider) override
    {
        const auto radius = std::min((height / 2.0f), (width / 2.0f));
        const auto centreX = x + width * 0.5f;
        const auto centreY = y + height * 0.5f;
        const auto rw = radius * 2.0f;
        const auto angle = rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);


        // Read the display scale. Clamp to [1, 4]. Fall back to 1.0 off-display.
        double scaleFactor = Component::getApproximateScaleFactorForComponent(&slider);
        if (scaleFactor < 1.0) scaleFactor = 1.0;
        if (scaleFactor > 4.0) scaleFactor = 4.0;

        // The cache key is the physical diameter.
        const int d = juce::roundToInt(rw);
        const int physicalD = juce::roundToInt(static_cast<double>(d) * scaleFactor);
        rebuildCache(physicalD);
        const float scale = static_cast<float>(d) / 431.0f;

        g.setImageResamplingQuality(Graphics::highResamplingQuality);

        // Draw a cached layer. The image is at the physical size,
        // so the transform scales down to the logical size and carries
        // rotation and translation.
        auto drawLayer = [&](const Image &img,
                                 const float cxSrc,
                                 const float cySrc,
                                 const float rotationAngle)
        {
            if (! img.isValid())
                return;
            const float cx = cxSrc * scale;
            const float cy = cySrc * scale;

            const float invScale = static_cast<float>(1.0 / scaleFactor);

            AffineTransform t;
            t = t.scaled(invScale);
            t = t.translated(-cx, -cy);
            if (rotationAngle != 0.0f) t = t.rotated(rotationAngle);
            t = t.translated(centreX, centreY);
            g.drawImageTransformed(img, t);
        };

        drawLayer(cache_.ellipse2, 215.0f, 157.5f, 0.0f);
        drawLayer(cache_.ellipse3,
                  static_cast<float>(srcEllipse3.getWidth()) * 0.5f,
                  static_cast<float>(srcEllipse3.getHeight()) * 0.5f, 0.0f);
        drawLayer(cache_.ellipse4,
                  static_cast<float>(srcEllipse4.getWidth()) * 0.5f,
                  static_cast<float>(srcEllipse4.getHeight()) * 0.5f, angle);

        if (cache_.rectangle2.isValid())
        {
            const float rcx = static_cast<float>(srcRectangle2.getWidth()) * 0.5f * scale;
            const float rcy = static_cast<float>(srcRectangle2.getHeight()) * 0.5f * scale;
            const float invScale = static_cast<float>(1.0 / scaleFactor);

            AffineTransform t;
            t = t.scaled(invScale);
            t = t.translated(-rcx, -rcy);
            t = t.translated(0.0f, -80.0f * scale);
            t = t.rotated(angle);
            t = t.translated(centreX, centreY);
            g.drawImageTransformed(cache_.rectangle2, t);
        }
    }

    private:
        struct CachedFrames {
            int diameter = 0;
            Image ellipse2;
            Image ellipse3;
            Image ellipse4;
            Image rectangle2;
        };

        void rebuildCache(const int diameter)
        {
            if (diameter == cache_.diameter)
                return;

            cache_.diameter = diameter;
            const float scale = static_cast<float>(diameter) / 431.0f;

            auto rescale = [&](const Image& src) -> Image
            {
                if (! src.isValid())
                    return {};
                return resampleTo(src,
                                   juce::roundToInt(static_cast<float>(src.getWidth()) * scale),
                                   juce::roundToInt(static_cast<float>(src.getHeight()) * scale));
            };

            cache_.ellipse2 = rescale(srcEllipse2);
            cache_.ellipse3 = rescale(srcEllipse3);
            cache_.ellipse4 = rescale(srcEllipse4);
            cache_.rectangle2 = rescale(srcRectangle2);
        }

        Image srcEllipse2;
        Image srcEllipse3;
        Image srcEllipse4;
        Image srcRectangle2;
        CachedFrames cache_;
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
                const float middlePos = 0.5f;
                const auto middleAngle = rotaryStartAngle + middlePos * (rotaryEndAngle - rotaryStartAngle);
                return {std::min(angle, middleAngle), std::max(angle, middleAngle)};
            }

            return {rotaryStartAngle, angle};
        }
    }

    class PDLKnob : public Component,
                    private Slider::Listener,
            private Timer
    {
    public:
        PDLKnob(const String &labelText,
                AudioProcessorValueTreeState &state,
                const ParameterID &pid);

        ~PDLKnob() override;

        void paint(Graphics &g) override;
        void resized() override;
        Slider &getSlider() { return slider; }
        void setLabelText(const String &text) { labelText_ = text; repaint(); }
        void setDrawFromMiddle(bool v) { slider.getProperties().set("drawFromMiddle", v); }

        // Store the accent colour for the value text.
        void setAccentColour(Colour c) { accentColour_ = c; repaint(); }

        // Set the tooltip, title, and help text on the slider.
        void setTooltip(const String &text)
        {
            slider.setTooltip(text);
            slider.setTitle(labelText_);
            slider.setHelpText(text);
        }

        // Mouse listener callbacks for the slider.
        void mouseEnter(const MouseEvent &e) override;
        void mouseExit(const MouseEvent &e) override;
        void mouseWheelMove(const MouseEvent &e, const MouseWheelDetails &wheel) override;
        void mouseDoubleClick(const MouseEvent &e) override;

    private:
        void sliderValueChanged(Slider *) override { repaint(); }
        void sliderDragStarted(Slider *) override;
        void sliderDragEnded(Slider *) override;
        void timerCallback() override;

        void startHoldTimer();
        void showValueEditor_();
        void applyEditorText_();

        Slider slider;
        String labelText_;
        Colour labelColour_ { Colours::textDim };
        Colour accentColour_ { Colours::accentDelayDigital };
        PedalKnob lnf;
        std::unique_ptr<AudioProcessorValueTreeState::SliderAttachment> attachment;
        AudioProcessorValueTreeState &apvtsRef_;
        String paramID_;
        Label valueEditor_;
        bool showValue_ = false;
        bool hovered_ = false;
        bool dragging_ = false;
        Time lastWheelTime_;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PDLKnob)
    };
}
#endif
