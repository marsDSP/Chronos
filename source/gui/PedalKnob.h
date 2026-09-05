#pragma once

#ifndef CHRONOS_PEDALKNOB_H
#define CHRONOS_PEDALKNOB_H

#include <JuceHeader.h>
#include "Colours.h"
#include "Fonts.h"
#include "Metrics.h"
#include <array>
#include <vector>
#include <limits>

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

    // The shared knob look and feel. One instance per editor serves every PDLKnob.
    // The frame cache holds up to kCacheCap entries keyed by physical diameter.
    // drawRotarySlider never resamples. On a miss it draws the nearest cached
    // entry under an adjusted transform and requests a rebuild. The rebuild
    // runs on a debounce timer and resamples the missing sizes once.
    class PedalKnob : public LookAndFeel_V4, private Timer {
    public:
        PedalKnob()
        {
            srcEllipse2 = ImageCache::getFromMemory(BinaryData::Ellipse_2_png, BinaryData::Ellipse_2_pngSize);
            srcEllipse3 = ImageCache::getFromMemory(BinaryData::Ellipse_3_png, BinaryData::Ellipse_3_pngSize);
            srcEllipse4 = ImageCache::getFromMemory(BinaryData::Ellipse_4_png, BinaryData::Ellipse_4_pngSize);
            srcRectangle2 = ImageCache::getFromMemory(BinaryData::Rectangle_2_png, BinaryData::Rectangle_2_pngSize);
        }

        ~PedalKnob() override
        {
            stopTimer();
        }

        // Register a slider for repaint after a cache rebuild.
        void registerSlider(Slider* s)
        {
            sliders_.push_back(s);
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
        const float scale = static_cast<float>(d) / 431.0f;

        // Find the exact or nearest cache entry.
        const CachedFrames* exact = nullptr;
        const CachedFrames* nearest = nullptr;
        int nearestDiff = std::numeric_limits<int>::max();

        for (const auto& entry : cache_)
        {
            if (entry.diameter == physicalD)
            {
                exact = &entry;
                break;
            }
            const int diff = std::abs(entry.diameter - physicalD);
            if (diff < nearestDiff)
            {
                nearestDiff = diff;
                nearest = &entry;
            }
        }

        // Fall back to the source images when the cache is empty.
        const int entryDiameter = exact ? exact->diameter
                              : (nearest ? nearest->diameter : 431);

        // Request a rebuild on a miss. Never resample in paint.
        if (exact == nullptr)
            requestRebuild_(physicalD);

        // The draw scale maps the drawn image pixels to logical pixels.
        const float drawScale = static_cast<float>(d) / static_cast<float>(entryDiameter);

        g.setImageResamplingQuality(Graphics::highResamplingQuality);

        // An inert knob draws every layer at the inert alpha.
        if (! slider.isEnabled())
            g.setOpacity(kInertAlpha);

        // Draw a cached layer. The image is at the entry diameter,
        // so the draw transform scales to the logical size and carries
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

            AffineTransform t;
            t = t.scaled(drawScale);
            t = t.translated(-cx, -cy);
            if (rotationAngle != 0.0f) t = t.rotated(rotationAngle);
            t = t.translated(centreX, centreY);
            g.drawImageTransformed(img, t);
        };

        const Image& imgEllipse2 = exact ? exact->ellipse2
                               : (nearest ? nearest->ellipse2 : srcEllipse2);
        const Image& imgEllipse3 = exact ? exact->ellipse3
                               : (nearest ? nearest->ellipse3 : srcEllipse3);
        const Image& imgEllipse4 = exact ? exact->ellipse4
                               : (nearest ? nearest->ellipse4 : srcEllipse4);
        const Image& imgRectangle2 = exact ? exact->rectangle2
                                : (nearest ? nearest->rectangle2 : srcRectangle2);

        drawLayer(imgEllipse2, 215.0f, 157.5f, 0.0f);
        drawLayer(imgEllipse3,
                  static_cast<float>(srcEllipse3.getWidth()) * 0.5f,
                  static_cast<float>(srcEllipse3.getHeight()) * 0.5f, 0.0f);
        drawLayer(imgEllipse4,
                  static_cast<float>(srcEllipse4.getWidth()) * 0.5f,
                  static_cast<float>(srcEllipse4.getHeight()) * 0.5f, angle);

        if (imgRectangle2.isValid())
        {
            const float rcx = static_cast<float>(srcRectangle2.getWidth()) * 0.5f * scale;
            const float rcy = static_cast<float>(srcRectangle2.getHeight()) * 0.5f * scale;

            AffineTransform t;
            t = t.scaled(drawScale);
            t = t.translated(-rcx, -rcy);
            t = t.translated(0.0f, -80.0f * scale);
            t = t.rotated(angle);
            t = t.translated(centreX, centreY);
            g.drawImageTransformed(imgRectangle2, t);
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

        static constexpr int kCacheCap = 6;
        static constexpr int kRebuildMs = 150;

        // Pending diameters to rebuild. Fixed-size: no allocation in paint.
        std::array<int, kCacheCap> pending_ {};
        int pendingCount_ = 0;

        std::vector<CachedFrames> cache_;
        std::vector<Component::SafePointer<Slider>> sliders_;

        void timerCallback() override
        {
            stopTimer();
            rebuildPending_();
        }

        // Add a diameter to the pending set and restart the debounce timer.
        void requestRebuild_(const int diameter)
        {
            for (int i = 0; i < pendingCount_; ++i)
                if (pending_[static_cast<std::size_t>(i)] == diameter)
                    return;

            if (pendingCount_ < kCacheCap)
            {
                pending_[static_cast<std::size_t>(pendingCount_)] = diameter;
                ++pendingCount_;
            }

            stopTimer();
            startTimer(kRebuildMs);
        }

        // Resample every pending diameter into the cache. Evict the least
        // recently used entry when the cache exceeds the cap. Repaint
        // every registered slider.
        void rebuildPending_()
        {
            for (int i = 0; i < pendingCount_; ++i)
            {
                const int diameter = pending_[static_cast<std::size_t>(i)];

                bool exists = false;
                for (const auto& entry : cache_)
                    if (entry.diameter == diameter) { exists = true; break; }
                if (exists) continue;

                CachedFrames entry;
                entry.diameter = diameter;
                const float s = static_cast<float>(diameter) / 431.0f;

                auto rescale = [&](const Image& src) -> Image
                {
                    if (! src.isValid())
                        return {};
                    return resampleTo(src,
                                       juce::roundToInt(static_cast<float>(src.getWidth()) * s),
                                       juce::roundToInt(static_cast<float>(src.getHeight()) * s));
                };

                entry.ellipse2 = rescale(srcEllipse2);
                entry.ellipse3 = rescale(srcEllipse3);
                entry.ellipse4 = rescale(srcEllipse4);
                entry.rectangle2 = rescale(srcRectangle2);

                cache_.insert(cache_.begin(), std::move(entry));

                while (static_cast<int>(cache_.size()) > kCacheCap)
                    cache_.pop_back();
            }

            pendingCount_ = 0;

            // Repaint all registered sliders. Drop dead pointers.
            for (auto& sp : sliders_)
                if (sp != nullptr)
                    sp->repaint();

            sliders_.erase(
                std::remove_if(sliders_.begin(), sliders_.end(),
                               [](const Component::SafePointer<Slider>& p) { return p == nullptr; }),
                sliders_.end());
        }

        Image srcEllipse2;
        Image srcEllipse3;
        Image srcEllipse4;
        Image srcRectangle2;
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
                const ParameterID &pid,
                PedalKnob& knobLnf);

        ~PDLKnob() override;

        void paint(Graphics &g) override;
        void resized() override;
        Slider &getSlider() { return slider; }
        void setLabelText(const String &text) { labelText_ = text; repaint(); }
        void setDrawFromMiddle(bool v) { slider.getProperties().set("drawFromMiddle", v); }

        // Store the accent colour for the value text.
        void setAccentColour(Colour c) { accentColour_ = c; repaint(); }

        // Store the scale metrics and relayout.
        void setMetrics(const Metrics& m) { metrics_ = m; resized(); repaint(); }

        // Set the tooltip and help text on the slider.
        void setTooltip(const String &text)
        {
            slider.setTooltip(text);
            slider.setHelpText(text);
        }

        // Mouse listener callbacks for the slider.
        void mouseEnter(const MouseEvent &e) override;
        void mouseExit(const MouseEvent &e) override;
        void mouseWheelMove(const MouseEvent &e, const MouseWheelDetails &wheel) override;
        void mouseDoubleClick(const MouseEvent &e) override;

        // Repaint at the new alpha when the enablement changes.
        void enablementChanged() override;

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
        Metrics metrics_;
        PedalKnob& lnfRef_;
        std::unique_ptr<AudioProcessorValueTreeState::SliderAttachment> attachment;
        AudioProcessorValueTreeState &apvtsRef_;
        String paramID_;
        Label valueEditor_;
        bool showValue_ = false;
        bool hovered_ = false;
        bool dragging_ = false;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PDLKnob)
    };
}
#endif
