#pragma once

#ifndef CHRONOS_BUMP_ARENA_H
#define CHRONOS_BUMP_ARENA_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

namespace MarsDSP::Memory {
    // ──────────────────────────────────────────────────────────────────────
    // BumpArena — one aligned allocation, bump-carved into typed spans.
    //
    // Chronos performs no runtime allocation: every buffer is sized in
    // prepare() and process() is allocation-free. The arena exists for
    // LAYOUT and INSTRUMENTATION, not speed: one 64-byte-aligned region the
    // chain's scratch spans are carved from contiguously (replacing N
    // separate heap regions the chain streamed across per block), one
    // allocation to reason about, and one get_total_num_bytes() figure for
    // the memory map.
    //
    // Invariants / contract:
    //  * reset(bytes) is PREPARE-TIME ONLY: it frees the old region and
    //    allocates a new one (operator new[] can throw bad_alloc; prepare is
    //    the only caller). The base is kBaseAlignment-aligned, so any carve
    //    with align <= kBaseAlignment stays aligned.
    //  * allocate_bytes is noexcept and returns nullptr on exhaustion. It
    //    NEVER throws, so it can appear in a noexcept path (cf. an STL-
    //    adapter arena that throws bad_alloc — unusable here).
    //  * clear() rewinds the cursor without freeing; re-serves the identical
    //    addresses in the identical order. Under CHRONOS_ARENA_DEBUG it also
    //    poisons the region (cheap at prepare; catches use-after-clear).
    //  * The cursor is stored BY VALUE. A reference cursor (e.g. a
    //    `size_t& used = used_` member) would copy-initialize from the
    //    source under the defaulted move constructor, so a moved-to arena's
    //    cursor would bind to the moved-from object's storage and dangle;
    //    the reference member would also implicitly delete move assignment.
    //    Move here transfers the pointer and zeroes the source, so
    //    move-construct-then-destroy-source is safe (regression-tested by
    //    bump_arena_check).
    //  * Allocation idiom matches Pow2RingBuffer: operator new[](bytes,
    //    std::align_val_t) + operator delete[](p, std::align_val_t) — one
    //    idiom across the codebase, no posix_memalign/_aligned_malloc.
    //  * JUCE-free (source/dsp/ and the harnesses link SharedCode only):
    //    local align_up_, assert, CHRONOS_ARENA_DEBUG instead of
    //    juce::snapPointerToAlignment / jassert / JUCE_DEBUG.
    //
    // Deliberately absent (no Chronos use): scoped Frame RAII (no scoped-
    // temporary allocation in the engine), a std::span view specialization,
    // and any growth path — all sizes are known at prepare, and growing on
    // demand would allocate on the audio thread.
    // ──────────────────────────────────────────────────────────────────────
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

        // prepare-time only; allocates. Frees any previous region. Not
        // noexcept: operator new[] may throw bad_alloc (prepare is the only
        // caller, never the audio thread).
        void reset(std::size_t bytes)
        {
            free_();
            if (bytes > 0)
            {
                data_ = static_cast<std::byte*>(
                    operator new[](bytes, std::align_val_t{ kBaseAlignment }));
                total_ = bytes;
            }
        }

        // Rewind the cursor, no free. Re-serves identical addresses in
        // identical order. CHRONOS_ARENA_DEBUG poisons first.
        void clear() noexcept
        {
#ifdef CHRONOS_ARENA_DEBUG
            if (data_ != nullptr)
                std::memset(data_, 0xDD, total_);
#endif
            used_ = 0;
        }

        // Bump-allocate n bytes at the requested power-of-two alignment
        // (<= kBaseAlignment). nullptr on exhaustion — never throws.
        [[nodiscard]] void* allocate_bytes(std::size_t n, std::size_t align) noexcept
        {
            assert(align > 0 && (align & (align - 1)) == 0 && "align must be a power of two");
            assert(align <= kBaseAlignment && "base region is only kBaseAlignment-aligned");
            const std::size_t start = align_up_(used_, align);
            if (data_ == nullptr || n > total_ - start)   // start <= used_ <= total_
                return nullptr;
            used_ = start + n;
            return data_ + start;
        }

        template<class T>
        [[nodiscard]] T* allocate(std::size_t n, std::size_t align = alignof(T)) noexcept
        {
            return static_cast<T*>(allocate_bytes(n * sizeof(T), align));
        }

        [[nodiscard]] std::size_t get_bytes_used() const noexcept { return used_; }
        [[nodiscard]] std::size_t get_total_num_bytes() const noexcept { return total_; }

    private:
        static std::size_t align_up_(std::size_t v, std::size_t align) noexcept
        {
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
