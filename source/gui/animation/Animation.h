#pragma once

#ifndef CHRONOS_ANIMATION_H
#define CHRONOS_ANIMATION_H

#include <algorithm>
#include <chrono>
#include <limits>

namespace MarsDSP::Animation {

inline float sin1(float phase) {
    phase = 0.5f - phase;
    const float phase2 = phase * phase;
    const float phase4 = phase2 * phase2;
    const float coefficient4 = phase4 * 12.228473185021549602f - phase2 *
                                        38.12119956657129365f +
                                        67.04364396354298358f;
    const float stage = coefficient4 * phase4 - phase2 * 64.834670562974805234f + 25.13273028802431777f;
    return stage * phase * (0.25f - phase2);
}

static constexpr int kFastTimeMs = 50;
static constexpr int kRegularTimeMs = 80;
static constexpr int kSlowTimeMs = 240;

enum EasingFunction {
    kLinear,
    kEaseIn,
    kEaseOut,
    kEaseInOut,
};

template<typename T>
class Animation {
public:
    Animation() : Animation(kRegularTimeMs, kEaseIn, kEaseOut) { }

    explicit Animation(int milliseconds, EasingFunction forwardEasing = kLinear,
                       EasingFunction backwardEasing = kLinear) :
        time_(milliseconds), forwardEasing_(forwardEasing),
        backwardEasing_(backwardEasing) { }

    static T interpolate(const T& from, const T& to, float t) { return from + (to - from) * t; }

    static T ease(const T& from, const T& to, float t, EasingFunction easing) {
        switch (easing) {
        case kEaseIn:  return interpolate(from, to, 1.0f - sin1(0.25f * (1.0f - t)));
        case kEaseOut: return interpolate(from, to, sin1(0.25f * t));
        case kEaseInOut: return interpolate(from, to, sin1(0.5f * t - 0.25f) * 0.5f + 0.5f);
        case kLinear:
        default: return interpolate(from, to, t);
        }
    }

    void target(bool targeting, bool jump = false) {
        lastMs_ = elapsedMs();
        targeting_ = targeting;
        if (jump)
            t_ = targeting ? 1.0f : 0.0f;
    }

    [[nodiscard]] bool isTargeting() const { return targeting_; }
    [[nodiscard]] bool isAnimating() const { return targeting_ ? t_ < 1.0f : t_ > 0.0f; }

    void setSourceValue(T value) { source_ = value; }
    void setTargetValue(T value) { target_ = value; }
    T sourceValue() const { return source_; }
    T targetValue() const { return target_; }
    void setAnimationTime(int milliseconds) { time_ = milliseconds; }

    T value() const {
        if (t_ <= 0.0f)
            return source_;

        float t = t_;
        EasingFunction easing = forwardEasing_;
        const T* from = &source_;
        const T* to = &target_;

        if (! targeting_) {
            easing = backwardEasing_;
            from = &target_;
            to = &source_;
            t = 1.0f - t_;
        }

        return ease(*from, *to, t, easing);
    }

    T update() {
        const long long ms = elapsedMs();
        float delta = 1.0f;
        if (time_ > std::numeric_limits<float>::epsilon())
            delta = static_cast<float>(ms - lastMs_) / static_cast<float>(time_);
        lastMs_ = ms;

        if (targeting_)
            t_ = std::min(t_ + delta, 1.0f);
        else
            t_ = std::max(t_ - delta, 0.0f);

        return value();
    }

private:
    static long long elapsedMs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    T source_ {};
    T target_ {};
    int time_ = kRegularTimeMs;
    long long lastMs_ = 0;

    EasingFunction forwardEasing_ = kLinear;
    EasingFunction backwardEasing_ = kLinear;

    bool targeting_ = false;
    float t_ = 0.0f;
};

}
#endif
