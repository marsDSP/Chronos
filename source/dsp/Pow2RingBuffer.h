#pragma once

#ifndef CHRONOS_POW2_RING_BUFFER_H
#define CHRONOS_POW2_RING_BUFFER_H

#include <bit>
#include <cassert>
#include <cstring>
#include <memory>
#include <new>
#include <algorithm>

namespace MarsDSP::Buffers {
    template<typename SampleType, int MirrorSamples = 32>
    class Pow2RingBuffer {
    public:
        static_assert(MirrorSamples > 0, "MirrorSamples must be positive");
        static_assert(MirrorSamples % 8 == 0, "MirrorSamples must be a multiple of 8");

        void prepare(int numChannels, int minimumCapacitySamples);

        void clear() noexcept;

        // ---- geometry ----
        int getCapacity() const noexcept;
        int getMask() const noexcept;
        int getNumChannels() const noexcept;
        int getWriteIndex() const noexcept;
        int wrap(int index) const noexcept;

        // ---- block write ----
        void writeAt(int channel, int startIndex, const SampleType *src, int numSamples) noexcept;
        void refreshMirror() noexcept;
        void advance(int numSamples) noexcept;

        // ---- window read ----
        void readWindow(int channel, int startIndex, SampleType *dst, int length) const noexcept;
        const SampleType *tryGetContiguous(int channel, int startIndex, int length) const noexcept;

        // ---- copy / move ----
        Pow2RingBuffer() noexcept = default;
        Pow2RingBuffer(const Pow2RingBuffer &) = delete;
        Pow2RingBuffer &operator=(const Pow2RingBuffer &) = delete;
        Pow2RingBuffer(Pow2RingBuffer &&) noexcept = default;
        Pow2RingBuffer &operator=(Pow2RingBuffer &&) noexcept = default;
    };
}
#endif
