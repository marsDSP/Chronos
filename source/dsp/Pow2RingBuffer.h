#pragma once

#ifndef CHRONOS_POW2_RING_BUFFER_H
#define CHRONOS_POW2_RING_BUFFER_H

#include "utils/memory/BumpArena.h"

#include <bit>
#include <cassert>
#include <cstddef>
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

        void prepare(const int minimumCapacity) noexcept
        {
            assert(minimumCapacity > 0);
            const auto rounded = std::bit_ceil(static_cast<unsigned int>(minimumCapacity));
            const auto newCapacity = static_cast<int>(rounded);
            assert(std::has_single_bit(static_cast<unsigned int>(newCapacity)));
            assert(newCapacity >= kTail);

            if (const int need = newCapacity + kTail; !storage_ || need > allocated_)
            {
                const auto bytes = static_cast<std::size_t>(need) * sizeof(float);
                const auto raw = operator new[](bytes, std::align_val_t{16});
                storage_.reset(static_cast<float *>(raw));
                allocated_ = need;
            }

            data_ = storage_.get();
            capacity_ = newCapacity;
            mask_ = newCapacity - 1;
            clear();
        }

        void prepare(const int minimumCapacity, Memory::BumpArena &arena) noexcept
        {
            assert(minimumCapacity > 0);
            const auto rounded = std::bit_ceil(static_cast<unsigned int>(minimumCapacity));
            const auto newCapacity = static_cast<int>(rounded);
            assert(std::has_single_bit(static_cast<unsigned int>(newCapacity)));
            assert(newCapacity >= kTail);

            const int need = newCapacity + kTail;
            storage_.reset(); // release any owned storage
            data_ = arena.allocate<float>(arenaFloatsFor(minimumCapacity), Memory::BumpArena::kBaseAlignment);
            assert(data_ != nullptr && "arena under-sized for this ring (see arenaFloatsFor)");
            allocated_ = need;
            capacity_ = newCapacity;
            mask_ = newCapacity - 1;
            clear();
        }

        static constexpr std::size_t arenaFloatsFor(const int minimumCapacity) noexcept
        {
            const auto rounded = std::bit_ceil(static_cast<unsigned int>(minimumCapacity));
            const std::size_t need = static_cast<std::size_t>(rounded) + kTail;
            return (need + 15u) & ~static_cast<std::size_t>(15u);
        }

        void clear() const noexcept
        {
            if (data_ != nullptr) std::memset(data_, 0, static_cast<std::size_t>(capacity_ + kTail) * sizeof(float));
        }

        [[nodiscard]] int getCapacity() const noexcept { return capacity_; }
        [[nodiscard]] int mask() const noexcept { return mask_; }

        void writeBlock(const float *src, const int startIdx, const int n) const noexcept
        {
            assert(startIdx >= 0 && startIdx < capacity_);
            assert(n > 0 && n <= capacity_);

            const int first = std::min(n, capacity_ - startIdx);
            std::memcpy(data_ + startIdx, src, static_cast<std::size_t>(first) * sizeof(float));
            const int remainder = n - first;
            if (remainder > 0) std::memcpy(data_, src + first, static_cast<std::size_t>(remainder) * sizeof(float));
        }

        void refreshMirror(const int startIdx, const int n) const noexcept
        {
            const bool wrapped = startIdx + n > capacity_;
            const bool touchedHead = startIdx < kTail;
            if (wrapped || touchedHead) std::memcpy(data_ + capacity_, data_, static_cast<std::size_t>(kTail) * sizeof(float));
        }

        void readWindow(float *dst, const int startIdx, const int len) const noexcept
        {
            assert(startIdx >= 0 && startIdx < capacity_);
            assert(len > 0 && len <= capacity_);

            const int first = std::min(len, capacity_ + kTail - startIdx);
            std::memcpy(dst, data_ + startIdx, static_cast<std::size_t>(first) * sizeof(float));
            const int remainder = len - first;
            if (remainder > 0) std::memcpy(dst + first, data_ + kTail, static_cast<std::size_t>(remainder) * sizeof(float));
        }

        [[nodiscard]] const float *windowPtr(int startIdx, int len) const noexcept
        {
            assert(startIdx >= 0 && startIdx < capacity_);
            assert(len > 0 && len <= capacity_);
            return (startIdx + len <= capacity_ + kTail) ? data_ + startIdx : nullptr;
        }

        Pow2RingBuffer() noexcept = default;
        Pow2RingBuffer(const Pow2RingBuffer &) = delete;
        Pow2RingBuffer &operator=(const Pow2RingBuffer &) = delete;
        Pow2RingBuffer(Pow2RingBuffer &&o) noexcept : storage_(std::move(o.storage_)), data_(o.data_),
                                                      allocated_(o.allocated_), capacity_(o.capacity_),
                                                      mask_(o.mask_)
        {
            o.data_ = nullptr;
            o.allocated_ = 0;
            o.capacity_ = 0;
            o.mask_ = 0;
        }

        Pow2RingBuffer &operator=(Pow2RingBuffer &&o) noexcept
        {
            if (this != &o)
            {
                storage_ = std::move(o.storage_);
                data_ = o.data_;
                allocated_ = o.allocated_;
                capacity_ = o.capacity_;
                mask_ = o.mask_;
                o.data_ = nullptr;
                o.allocated_ = 0;
                o.capacity_ = 0;
                o.mask_ = 0;
            }
            return *this;
        }

    private:
        struct Deleter
        {
            void operator()(float *p) const noexcept
            {
                if (p) operator delete[](p, std::align_val_t{16});
            }
        };

        using Ptr = std::unique_ptr<float[], Deleter>;

        Ptr storage_; // owning (null when arena-backed)
        float *data_ = nullptr; // active storage: storage_.get() or arena carve
        int allocated_ = 0;
        int capacity_ = 0;
        int mask_ = 0;
    };
}
#endif
