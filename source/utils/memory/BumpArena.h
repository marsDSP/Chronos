#pragma once

#ifndef CHRONOS_BUMP_ARENA_H
#define CHRONOS_BUMP_ARENA_H

#include <cassert>
#include <cstddef>
#include <bit>
#include <cstring>
#include <new>

namespace MarsDSP::Memory {
    class BumpArena {
    public:
        static constexpr std::size_t kBaseAlignment = 64;

        BumpArena() noexcept = default;
        explicit BumpArena(std::size_t bytes) { reset(bytes); }
        ~BumpArena() { free_(); }

        BumpArena(const BumpArena&) = delete;
        BumpArena& operator=(const BumpArena&) = delete;

        BumpArena(BumpArena&& other) noexcept { moveFrom_(other); }
        BumpArena& operator=(BumpArena&& other) noexcept
        {
            if (this != &other)
            {
                free_();
                moveFrom_(other);
            }
            return *this;
        }

        void reset(std::size_t bytes)
        {
            free_();
            if (bytes > 0)
            {
                data_ = static_cast<std::byte*>(operator new[](bytes, std::align_val_t{ kBaseAlignment }));
                total_ = bytes;
            }
        }

        void clear() noexcept
        {
#ifdef CHRONOS_ARENA_DEBUG
            if (data_ != nullptr)
                std::memset(data_, 0xDD, total_);
#endif
            used_ = 0;
        }

        [[nodiscard]] std::byte* allocate_bytes(std::size_t n, std::size_t align) noexcept
        {
            assert(align > 0 && (align & (align - 1)) == 0 && "align must be a power of two");
            assert(align <= kBaseAlignment && "base region is only kBaseAlignment-aligned");
            const std::size_t start = align_up_(used_, align);
            if (data_ == nullptr || n > total_ - start)   // start <= used_ <= total_
                return nullptr;
            used_ = start + n;
            return data_ + start;
        }

        // The arena aligns the base to kBaseAlignment. This exceeds
        // alignof(T) for every supported type. reinterpret_cast converts
        // std::byte* to T*.
        template<class T>
        [[nodiscard]] T* allocate(std::size_t n, std::size_t align = alignof(T)) noexcept
        {
            return std::bit_cast<T*>(allocate_bytes(n * sizeof(T), align));
        }

        [[nodiscard]] std::size_t get_bytes_used() const noexcept { return used_; }
        [[nodiscard]] std::size_t get_total_num_bytes() const noexcept { return total_; }

    private:
        static std::size_t align_up_(std::size_t v, std::size_t align) noexcept
        {
            // can theoretically overflow;
            // assert align - 1 > SIZE_MAX - v ?
            return (v + align - 1) & ~(align - 1);
        }

        void free_() noexcept
        {
            if (data_ != nullptr) operator delete[](data_, std::align_val_t{ kBaseAlignment });
            data_ = nullptr;
            total_ = 0;
            used_ = 0;
        }

        void moveFrom_(BumpArena& o) noexcept
        {
            data_ = o.data_;
            total_ = o.total_;
            used_ = o.used_;
            o.data_ = nullptr;
            o.total_ = 0;
            o.used_ = 0;
        }

        std::byte* data_ = nullptr;
        std::size_t total_ = 0;
        std::size_t used_ = 0;
    };
}
#endif
