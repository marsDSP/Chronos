#pragma once

#ifndef CHRONOS_SAMPLE_FIFO_H
#define CHRONOS_SAMPLE_FIFO_H

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>

namespace MarsDSP::Memory
{
    // A single-producer single-consumer float ring for block transfers.
    // The audio thread writes blocks, the message thread reads them. A
    // write that does not fit is dropped whole: the feed is best effort,
    // like the tap FIFO, and never blocks or allocates. The indices are
    // monotonic counters masked on access, so full and empty stay distinct.
    template<std::size_t CapacityPow2>
    class SampleFifo
    {
        static_assert(CapacityPow2 >= 2 && (CapacityPow2 & (CapacityPow2 - 1)) == 0,
                      "SampleFifo: CapacityPow2 must be a power of two >= 2");

    public:
        static constexpr std::size_t kCapacity = CapacityPow2;
        static constexpr std::size_t kMask = CapacityPow2 - 1;

        SampleFifo() noexcept = default;
        SampleFifo(const SampleFifo &) = delete;
        SampleFifo &operator=(const SampleFifo &) = delete;

        // Producer. Write n samples, or nothing when they do not all fit.
        // Return true when written.
        bool write(const float *src, const std::size_t n) noexcept
        {
            if (n == 0 || n > kCapacity) return n == 0;
            const std::size_t w = writeIndex_.load(std::memory_order_relaxed);
            const std::size_t r = readIndex_.load(std::memory_order_acquire);
            if (w - r + n > kCapacity)
                return false;

            const std::size_t start = w & kMask;
            const std::size_t first = std::min(n, kCapacity - start);
            std::memcpy(buffer_.data() + start, src, first * sizeof(float));
            if (first < n)
                std::memcpy(buffer_.data(), src + first, (n - first) * sizeof(float));

            writeIndex_.store(w + n, std::memory_order_release);
            return true;
        }

        // Consumer. Read up to maxCount samples. Return the count read.
        std::size_t read(float *dst, const std::size_t maxCount) noexcept
        {
            const std::size_t r = readIndex_.load(std::memory_order_relaxed);
            const std::size_t w = writeIndex_.load(std::memory_order_acquire);
            const std::size_t n = std::min(maxCount, w - r);
            if (n == 0) return 0;

            const std::size_t start = r & kMask;
            const std::size_t first = std::min(n, kCapacity - start);
            std::memcpy(dst, buffer_.data() + start, first * sizeof(float));
            if (first < n)
                std::memcpy(dst + first, buffer_.data(), (n - first) * sizeof(float));

            readIndex_.store(r + n, std::memory_order_release);
            return n;
        }

        // Consumer. The number of samples waiting.
        [[nodiscard]] std::size_t available() const noexcept
        {
            return writeIndex_.load(std::memory_order_acquire) - readIndex_.load(std::memory_order_relaxed);
        }

        // Consumer. Drop everything waiting.
        void clear() noexcept
        {
            readIndex_.store(writeIndex_.load(std::memory_order_acquire), std::memory_order_release);
        }

    private:
        static constexpr std::size_t kCacheLineBytes = 128;
        alignas(kCacheLineBytes) std::atomic<std::size_t> writeIndex_{0};
        alignas(kCacheLineBytes) std::atomic<std::size_t> readIndex_{0};
        alignas(kCacheLineBytes) std::array<float, CapacityPow2> buffer_{};
    };
} // namespace MarsDSP::Memory

#endif
