#pragma once

#ifndef CHRONOS_SEQ_MAKIMA_H
#define CHRONOS_SEQ_MAKIMA_H

// Modified-Akima ("makima") spline interpolator, vendored from the
// ProQClone reference (Source/seq_makima.hpp), itself vendored from the
// Autocrat2 / MarsDSP project. The spectrum display evaluates it once per
// pixel column over the FFT bin knots, so the trace curves smoothly between
// bins instead of kinking at each one. Local adaptations: the namespace,
// the include guard, a const evaluation input, and a guard in the weight
// so a run of equal knots (a silent trace) yields a zero slope, not 0/0.
//
// The knot arrays are referenced, not copied: they must outlive the
// interpolator and keep their addresses. prepare() reads the current knot
// values; eval() takes ascending x and clamps outside the knot range.

#include <cmath>
#include <cstddef>
#include <vector>

namespace MarsDSP::Interpolation {

template <typename FloatType>
class SeqMakima {
public:
    explicit SeqMakima(const FloatType* x, const FloatType* y, const size_t point_num,
                       const FloatType left_derivative, const FloatType right_derivative)
        : xs_(x), ys_(y), input_size_(point_num),
          left_derivative_(left_derivative), right_derivative_(right_derivative)
    {
        derivatives_.resize(point_num);
        deltas_.resize(point_num - 1);
    }

    void prepare()
    {
        for (size_t i = 0; i < deltas_.size(); ++i)
            deltas_[i] = (ys_[i + 1] - ys_[i]) / (xs_[i + 1] - xs_[i]);

        const auto left_delta = FloatType(2) * deltas_[0] - deltas_[1];
        const auto right_delta = FloatType(2) * deltas_[deltas_.size() - 1] - deltas_[deltas_.size() - 2];

        const auto n = derivatives_.size();
        derivatives_[0] = left_derivative_;
        derivatives_[n - 1] = right_derivative_;
        derivatives_[1] = calculateD(left_delta, deltas_[0], deltas_[1], deltas_[2]);
        for (size_t i = 2; i < n - 2; ++i)
            derivatives_[i] = calculateD(deltas_[i - 2], deltas_[i - 1], deltas_[i], deltas_[i + 1]);
        derivatives_[n - 2] = calculateD(deltas_[n - 4], deltas_[n - 3], deltas_[n - 2], right_delta);
    }

    void eval(const FloatType* x, FloatType* y, const size_t point_num) const
    {
        size_t current_pos = 0;
        size_t start_idx = 0, end_idx = point_num - 1;
        while (start_idx <= end_idx && x[start_idx] <= xs_[0])
        {
            y[start_idx] = ys_[0];
            start_idx += 1;
        }
        while (end_idx > start_idx && x[end_idx] >= xs_[input_size_ - 1])
        {
            y[end_idx] = ys_[input_size_ - 1];
            end_idx -= 1;
        }
        for (size_t i = start_idx; i <= end_idx; ++i)
        {
            while (current_pos + 2 < input_size_ && x[i] >= xs_[current_pos + 1])
                current_pos += 1;
            const auto t = (x[i] - xs_[current_pos]) / (xs_[current_pos + 1] - xs_[current_pos]);
            y[i] = h00(t) * ys_[current_pos]
                 + h10(t) * (xs_[current_pos + 1] - xs_[current_pos]) * derivatives_[current_pos]
                 + h01(t) * ys_[current_pos + 1]
                 + h11(t) * (xs_[current_pos + 1] - xs_[current_pos]) * derivatives_[current_pos + 1];
        }
    }

private:
    const FloatType* xs_;
    const FloatType* ys_;
    size_t input_size_;
    std::vector<FloatType> derivatives_, deltas_;
    FloatType left_derivative_, right_derivative_;

    static FloatType h00(const FloatType t) { return (FloatType(1) + FloatType(2) * t) * (FloatType(1) - t) * (FloatType(1) - t); }
    static FloatType h10(const FloatType t) { return t * (FloatType(1) - t) * (FloatType(1) - t); }
    static FloatType h01(const FloatType t) { return t * t * (FloatType(3) - FloatType(2) * t); }
    static FloatType h11(const FloatType t) { return t * t * (t - FloatType(1)); }

    static FloatType calculateD(const FloatType delta0, const FloatType delta1, const FloatType delta2, const FloatType delta3)
    {
        const auto w1 = std::abs(delta3 - delta2) + std::abs(delta3 + delta2) * FloatType(0.5);
        const auto w2 = std::abs(delta1 - delta0) + std::abs(delta1 + delta0) * FloatType(0.5);
        const auto sum = w1 + w2;
        if (! (sum > FloatType(0)))
            return FloatType(0.5) * (delta1 + delta2);
        const auto w = w1 / sum;
        return w * delta1 + (FloatType(1) - w) * delta2;
    }
};

} // namespace MarsDSP::Interpolation

#endif
