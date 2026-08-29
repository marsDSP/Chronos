#pragma once

#ifndef CHRONOS_SPSC_FIFO_H
#define CHRONOS_SPSC_FIFO_H

#include <array>
#include <atomic>
#include <cstddef>
#include <type_traits>

namespace MarsDSP::Memory
{
    // A single-producer single-consumer lock-free FIFO queue.
    // The queue uses a fixed power-of-two capacity array.
    // The producer pushes items.
    // The consumer pops items.
    template<typename T, std::size_t CapacityPow2>
    class SpscFifo
    {
        static_assert(std::is_trivially_copyable_v<T>,
                      "SpscFifo: T must be trivially copyable");
        static_assert(CapacityPow2 >= 2 && (CapacityPow2 & (CapacityPow2 - 1)) == 0,
                      "SpscFifo: CapacityPow2 must be a power of two >= 2");

    public:
        static constexpr std::size_t kCacheLineBytes = 128;
        static constexpr std::size_t kCapacity = CapacityPow2;
        static constexpr std::size_t kMask = CapacityPow2 - 1;

        SpscFifo() noexcept = default;

        ~SpscFifo() = default;

        SpscFifo(const SpscFifo &) = delete;

        SpscFifo &operator=(const SpscFifo &) = delete;

        SpscFifo(SpscFifo &&) = delete;

        SpscFifo &operator=(SpscFifo &&) = delete;

        // Push one item into the queue.
        // The producer thread must call this function.
        // Return false when the queue is full.
        [[nodiscard]] bool push(const T &element) noexcept
        {
            const std::size_t w = writeIndex_.load(std::memory_order_relaxed);
            if (w - cachedReadIndex_ >= CapacityPow2)
            {
                cachedReadIndex_ = readIndex_.load(std::memory_order_acquire);
                if (w - cachedReadIndex_ >= CapacityPow2)
                    return false;
            }

            buffer_[w & kMask] = element;
            writeIndex_.store(w + 1, std::memory_order_release);
            return true;
        }

        // Pop one item from the queue.
        // The consumer thread must call this function.
        // Return false when the queue is empty.
        [[nodiscard]] bool pop(T &result) noexcept
        {
            const std::size_t r = readIndex_.load(std::memory_order_relaxed);
            if (r == cachedWriteIndex_)
            {
                cachedWriteIndex_ = writeIndex_.load(std::memory_order_acquire);
                if (r == cachedWriteIndex_)
                    return false;
            }

            result = buffer_[r & kMask];
            readIndex_.store(r + 1, std::memory_order_release);
            return true;
        }

        // Reset the queue state.
        // The consumer thread must call this function.
        // Do not call this function while the producer pushes items.
        void clear() noexcept
        {
            const std::size_t w = writeIndex_.load(std::memory_order_relaxed);
            readIndex_.store(w, std::memory_order_relaxed);
            cachedWriteIndex_ = w;
        }

        [[nodiscard]] static constexpr std::size_t capacity() noexcept
        {
            return CapacityPow2;
        }

    private:
        alignas(kCacheLineBytes) std::atomic<std::size_t> writeIndex_{0};
        std::size_t cachedReadIndex_{0};
        char pad0_[kCacheLineBytes - sizeof(std::atomic<std::size_t>) - sizeof(std::size_t)]{};

        alignas(kCacheLineBytes) std::atomic<std::size_t> readIndex_{0};
        std::size_t cachedWriteIndex_{0};
        char pad1_[kCacheLineBytes - sizeof(std::atomic<std::size_t>) - sizeof(std::size_t)]{};

        alignas(kCacheLineBytes) std::array<T, CapacityPow2> buffer_{};
    };
} // namespace MarsDSP::Memory
#endif
