#pragma once

#ifndef CHRONOS_POW2_RING_BUFFER_H
#define CHRONOS_POW2_RING_BUFFER_H

#include <bit>
#include <cassert>
#include <cstring>
#include <memory>
#include <new>
#include <algorithm>

namespace MarsDSP::Delays
{
    class Pow2RingBuffer
    {
    public:
        static constexpr int kTail = 8;

        void prepare(int minimumCapacity) noexcept
        {
            assert(minimumCapacity > 0);
            const auto rounded = std::bit_ceil(static_cast<unsigned int>(minimumCapacity));
            const auto newCapacity = static_cast<int>(rounded);
            assert(std::has_single_bit(static_cast<unsigned int>(newCapacity)));
            assert(newCapacity >= kTail);

            if (const int need = newCapacity + kTail; !storage_ || need > allocated_)
            {
                const auto bytes = static_cast<size_t>(need) * sizeof(float);
                const auto raw = operator new[](bytes, std::align_val_t{16});
                storage_.reset(static_cast<float *>(raw));
                allocated_ = need;
            }
            capacity_ = newCapacity;
            mask_ = newCapacity - 1;
            clear();
        }

        void clear() noexcept
        {
            if (storage_) std::memset(storage_.get(), 0, static_cast<size_t>(capacity_ + kTail) * sizeof(float));
        }

        [[nodiscard]] int getCapacity() const noexcept { return capacity_; }
        [[nodiscard]] int mask() const noexcept { return mask_; }

        void writeBlock(const float *src, int startIdx, int n) noexcept
        {
            assert(startIdx >= 0 && startIdx < capacity_);
            assert(n > 0 && n <= capacity_);

            const int first = std::min(n, capacity_ - startIdx);
            std::memcpy(storage_.get() + startIdx, src, static_cast<size_t>(first) * sizeof(float));
            const int remainder = n - first;
            if (remainder > 0) std::memcpy(storage_.get(), src + first, static_cast<size_t>(remainder) * sizeof(float));
        }

        void refreshMirror(int startIdx, int n) noexcept
        {
            const bool wrapped = startIdx + n > capacity_;
            const bool touchedHead = startIdx < kTail;
            if (wrapped || touchedHead)
                std::memcpy(storage_.get() + capacity_, storage_.get(), static_cast<size_t>(kTail) * sizeof(float));
        }

        void readWindow(float *dst, int startIdx, int len) const noexcept
        {
            assert(startIdx >= 0 && startIdx < capacity_);
            assert(len > 0 && len <= capacity_);

            const int first = std::min(len, capacity_ + kTail - startIdx);
            std::memcpy(dst, storage_.get() + startIdx, static_cast<size_t>(first) * sizeof(float));
            const int remainder = len - first;
            if (remainder > 0)
                std::memcpy(dst + first, storage_.get() + kTail,
                            static_cast<size_t>(remainder) * sizeof(float));
        }

        // ---- copy / move ----
        Pow2RingBuffer() noexcept = default;
        Pow2RingBuffer(const Pow2RingBuffer &) = delete;
        Pow2RingBuffer &operator=(const Pow2RingBuffer &) = delete;
        Pow2RingBuffer(Pow2RingBuffer &&) noexcept = default;
        Pow2RingBuffer &operator=(Pow2RingBuffer &&) noexcept = default;

    private:
        struct Deleter
        {
            void operator()(float *p) const noexcept
            {
                if (p) ::operator delete[](p, std::align_val_t{16});
            }
        };

        using Ptr = std::unique_ptr<float[], Deleter>;

        Ptr storage_;
        int allocated_ = 0;
        int capacity_ = 0;
        int mask_ = 0;
    };
}
#endif
