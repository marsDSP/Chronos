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

        void prepare(int numChannels, int minimumCapacitySamples)
        {
            assert(numChannels > 0);
            assert(minimumCapacitySamples > 0);

            // round up to pow2 and pin the storage invariants
            const auto rounded = std::bit_ceil(static_cast<unsigned int>(minimumCapacitySamples));
            const int newCapacity = static_cast<int>(rounded);
            assert(std::has_single_bit(static_cast<unsigned int>(newCapacity)));
            assert(newCapacity >= kMirrorSamples);

            const int newStride = roundUpToMultipleOf8(newCapacity + kMirrorSamples);
            const auto needElements = static_cast<size_t>(numChannels) * static_cast<size_t>(newStride);

            // idempotent alloc
            if (!storage_ || needElements > allocatedElements_)
            {
                const auto bytes = needElements * sizeof(SampleType);
                const auto raw = operator new[](bytes, std::align_val_t{32});
                storage_.reset(static_cast<SampleType *>(raw));
                allocatedElements_ = needElements;
            }

            numChannels_ = numChannels;
            capacity_ = newCapacity;
            mask_ = newCapacity - 1;
            stride_ = newStride;
            writeIndex_ = 0;
            zeroStorage();
        }

        void clear() noexcept
        {
            writeIndex_ = 0;
            zeroStorage();
        }

        // ---- geometry ----
        int getCapacity() const noexcept { return capacity_; }
        int getMask() const noexcept { return mask_; }
        int getNumChannels() const noexcept { return numChannels_; }
        int getWriteIndex() const noexcept { return writeIndex_; }
        int wrap(int index) const noexcept { return index & mask_; }

        // ---- block write ----
        void writeAt(int channel, int startIndex, const SampleType *src, int numSamples) noexcept
        {
            assert(channel >= 0 && channel < numChannels_);
            assert(startIndex >= 0 && startIndex < capacity_);
            assert(numSamples > 0 && numSamples <= capacity_);

            SampleType *base = channelBase(channel);
            const int first = std::min(numSamples, capacity_ - startIndex);
            std::memcpy(base + startIndex, src, static_cast<size_t>(first) * sizeof(SampleType));
            const int remainder = numSamples - first;
            if (remainder > 0)
                std::memcpy(base, src + first, static_cast<size_t>(remainder) * sizeof(SampleType));
        }

        void refreshMirror() noexcept
        {
            if (!storage_) return;
            for (int ch = 0; ch < numChannels_; ++ch)
            {
                SampleType *base = channelBase(ch);
                std::memcpy(base + capacity_, base, static_cast<size_t>(kMirrorSamples) * sizeof(SampleType));
            }
        }

        void advance(int numSamples) noexcept
        {
            assert(numSamples >= 0);
            writeIndex_ = wrap(writeIndex_ + numSamples);
        }

        // ---- window read ----
        void readWindow(int channel, int startIndex, SampleType *dst, int length) const noexcept
        {
            assert(channel >= 0 && channel < numChannels_);
            assert(startIndex >= 0 && startIndex < capacity_);
            assert(length > 0 && length <= capacity_);

            const SampleType *src = tryGetContiguous(channel, startIndex, length);
            if (src)
            {
                std::memcpy(dst, src, static_cast<size_t>(length) * sizeof(SampleType));
                return;
            }

            const SampleType *base = channelBase(channel);
            const int first = std::min(length, capacity_ + kMirrorSamples - startIndex);
            std::memcpy(dst, base + startIndex, static_cast<size_t>(first) * sizeof(SampleType));
            const int remainder = length - first;
            if (remainder > 0)
                std::memcpy(dst + first, base + kMirrorSamples, static_cast<size_t>(remainder) * sizeof(SampleType));
        }

        const SampleType *tryGetContiguous(int channel, int startIndex, int length) const noexcept
        {
            assert(channel >= 0 && channel < numChannels_);
            assert(startIndex >= 0 && startIndex < capacity_);
            assert(length > 0);
            if (startIndex + length <= capacity_ + kMirrorSamples)
                return channelBase(channel) + startIndex;
            return nullptr;
        }

        // ---- block convenience ----
        int pushBlock(const SampleType *const *channelData, int numChannels, int numSamples) noexcept
        {
            assert(numChannels == numChannels_);
            assert(numSamples > 0 && numSamples <= capacity_);
            const int startIndex = writeIndex_;
            for (int ch = 0; ch < numChannels_; ++ch)
                writeAt(ch, startIndex, channelData[ch], numSamples);
            refreshMirror();
            advance(numSamples);
            return startIndex;
        }

        // ---- copy / move ----
        Pow2RingBuffer() noexcept = default;
        Pow2RingBuffer(const Pow2RingBuffer &) = delete;
        Pow2RingBuffer &operator=(const Pow2RingBuffer &) = delete;
        Pow2RingBuffer(Pow2RingBuffer &&) noexcept = default;
        Pow2RingBuffer &operator=(Pow2RingBuffer &&) noexcept = default;

    private:
        struct AlignedDeleter
        {
            void operator()(SampleType *p) const noexcept
            {
                if (p) operator delete[](static_cast<void *>(p), std::align_val_t{32});
            }
        };
        using AlignedPtr = std::unique_ptr<SampleType[], AlignedDeleter>;

        static constexpr int kMirrorSamples = MirrorSamples;

        static constexpr int roundUpToMultipleOf8(int n) noexcept { return n + 7 & ~7; }

        SampleType *channelBase(int channel) noexcept
        {
            return storage_.get() + static_cast<size_t>(channel) * static_cast<size_t>(stride_);
        }
        const SampleType *channelBase(int channel) const noexcept
        {
            return storage_.get() + static_cast<size_t>(channel) * static_cast<size_t>(stride_);
        }

        void zeroStorage() noexcept
        {
            if (!storage_) return;
            const auto bytes = static_cast<size_t>(numChannels_) * static_cast<size_t>(stride_) * sizeof(SampleType);
            std::memset(storage_.get(), 0, bytes);
        }

        // ---- storage + state ----
        AlignedPtr storage_;
        size_t allocatedElements_ = 0;
        int numChannels_ = 0;
        int capacity_ = 0;
        int mask_ = 0;
        int stride_ = 0;
        int writeIndex_ = 0;
    };
}
#endif
