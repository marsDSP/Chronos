/**
 * BumpArena correctness: per-carve alignment, exhaustion without throw,
 * clear() rewind re-serving identical addresses, byte accounting against a
 * hand-computed layout, move semantics without dangling, and edge cases.
 */

#include "utils/memory/BumpArena.h"

#include <cstdint>
#include <cstdio>
#include <print>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace
{
    using MarsDSP::Memory::BumpArena;

    const char *g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

    std::uintptr_t addr(const void *p) { return reinterpret_cast<std::uintptr_t>(p); }

    // (1) alignment
    void testAlignment()
    {
        g_section = "alignment";
        BumpArena a;
        a.reset(4096);

        const std::array<std::size_t, 5> aligns { { 4, 8, 16, 32, 64 } };
        const std::array<std::size_t, 7> sizes { { 1, 3, 7, 13, 16, 24, 100 } };
        for (std::size_t al: aligns)
            for (std::size_t n: sizes)
            {
                void *p = a.allocate_bytes(n, al);
                CHECK(p != nullptr);
                CHECK(addr(p) % al == 0);
            }
        std::println("alignment (odd sizes, aligns 4..64): PASS");
    }

    // (2) exhaustion
    void testExhaustion()
    {
        g_section = "exhaustion";
        BumpArena a;
        a.reset(256);

        void *p = a.allocate_bytes(200, 8);
        CHECK(p != nullptr);
        CHECK(a.get_bytes_used() == 200);

        // 100 bytes do not fit in the remaining 56: nullptr, cursor unmoved.
        CHECK(a.allocate_bytes(100, 8) == nullptr);
        CHECK(a.get_bytes_used() == 200);

        // 56 bytes fit exactly.
        void *q = a.allocate_bytes(56, 8);
        CHECK(q != nullptr);
        CHECK(a.get_bytes_used() == 256);

        // Full: even 1 byte fails. Alignment padding can also exhaust: a 64-
        // aligned 1-byte carve from a full arena has no room to pad into.
        CHECK(a.allocate_bytes(1, 8) == nullptr);
        CHECK(a.allocate_bytes(1, 64) == nullptr);
        CHECK(a.get_bytes_used() == 256);
        std::println("exhaustion (nullptr, no throw, cursor unmoved): PASS");
    }

    // (3) clear() rewind re-serves identical addresses
    void testClearRewind()
    {
        g_section = "clear-rewind";
        BumpArena a;
        a.reset(1024);

        constexpr int kSeq = 6;
        const std::array<std::size_t, kSeq> sizes = {{ 13, 40, 1, 128, 7, 200 }};
        const std::array<std::size_t, kSeq> aligns = {{ 8, 16, 4, 64, 8, 32 }};
        void *first[kSeq] = {};

        for (int i = 0; i < kSeq; ++i)
        {
            first[i] = a.allocate_bytes(sizes[i], aligns[i]);
            CHECK(first[i] != nullptr);
        }

        a.clear();
        CHECK(a.get_bytes_used() == 0);
        CHECK(a.get_total_num_bytes() == 1024);

        for (int i = 0; i < kSeq; ++i)
        {
            void *p = a.allocate_bytes(sizes[i], aligns[i]);
            CHECK(p == first[i]);
        }
        std::println("clear() rewind re-serves identical addresses: PASS");
    }

    // (4) byte accounting matches hand-computed layout
    void testByteAccounting()
    {
        g_section = "byte-accounting";
        BumpArena a;
        a.reset(1024);
        CHECK(a.get_bytes_used() == 0);
        CHECK(a.get_total_num_bytes() == 1024);

        // Hand-computed: used <- align_up(used, align) + n.
        void *p = a.allocate_bytes(13, 8); // align_up(0,8)=0   -> used 13
        CHECK(p != nullptr);
        CHECK(a.get_bytes_used() == 13);
        p = a.allocate_bytes(20, 16); // align_up(13,16)=16 -> used 36
        CHECK(p != nullptr);
        CHECK(a.get_bytes_used() == 36);
        p = a.allocate_bytes(1, 64); // align_up(36,64)=64 -> used 65
        CHECK(p != nullptr);
        CHECK(a.get_bytes_used() == 65);
        p = a.allocate_bytes(0, 8); // zero-size carve: pads, adds nothing
        CHECK(p != nullptr);
        CHECK(a.get_bytes_used() == 72);

        std::println("byte accounting (hand-computed layout): PASS");
    }

    // (5) move: construct-then-destroy-source, then move-assign
    void testMove()
    {
        g_section = "move";
        void *saved = nullptr;
        constexpr std::size_t kBytes = 512;

        BumpArena b; {
            BumpArena a;
            a.reset(kBytes);
            saved = a.allocate_bytes(64, 16);
            CHECK(saved != nullptr);
            std::memset(saved, 0xAB, 64); // pattern into the storage

            b = std::move(a); // move-ASSIGN; source empties
            CHECK(a.get_bytes_used() == 0);
            CHECK(a.get_total_num_bytes() == 0);
            CHECK(a.allocate_bytes(8, 8) == nullptr);
        } // source destroyed here

        // The moved-to arena still owns the (intact) storage.
        CHECK(b.get_total_num_bytes() == kBytes);
        CHECK(static_cast<const unsigned char*>(saved)[0] == 0xAB);
        CHECK(static_cast<const unsigned char*>(saved)[63] == 0xAB);

        // Move-CONSTRUCT, then destroy the source.
        BumpArena d; {
            BumpArena c = std::move(b); // move-CONSTRUCT
            CHECK(b.get_total_num_bytes() == 0);
            CHECK(c.get_total_num_bytes() == kBytes);
            CHECK(static_cast<const unsigned char*>(saved)[32] == 0xAB);
            d = std::move(c);
        }
        CHECK(d.get_total_num_bytes() == kBytes);
        CHECK(static_cast<const unsigned char*>(saved)[32] == 0xAB);
        std::println("move construct/assign + destroy-source (no dangle): PASS");
    }

    // (6) edges: default-constructed, typed allocate, base alignment
    void testEdges()
    {
        g_section = "edges";
        BumpArena a; // never reset
        CHECK(a.get_bytes_used() == 0);
        CHECK(a.get_total_num_bytes() == 0);
        CHECK(a.allocate_bytes(1, 8) == nullptr);
        CHECK(a.allocate<float>(4) == nullptr);

        BumpArena b;
        b.reset(256);
        float *f = b.allocate<float>(16); // 64 bytes
        CHECK(f != nullptr);
        CHECK(addr(f) % alignof(float) == 0);
        CHECK(b.get_bytes_used() == 64);
        for (int i = 0; i < 16; ++i) f[i] = static_cast<float>(i);
        CHECK(f[15] == 15.0f);

        // base region itself is kBaseAlignment-aligned.
        void *base = b.allocate_bytes(1, BumpArena::kBaseAlignment);
        CHECK(addr(base) % BumpArena::kBaseAlignment == 0);

        // reset() re-allocates (prepare-time): old extents forgotten.
        b.reset(128);
        CHECK(b.get_bytes_used() == 0);
        CHECK(b.get_total_num_bytes() == 128);
        std::println("edges (default arena, typed carve, base alignment): PASS");
    }
} // namespace

int main()
{
    std::println("=== Chronos bump_arena_check (C9) ===\n");
    testAlignment();
    testExhaustion();
    testClearRewind();
    testByteAccounting();
    testMove();
    testEdges();
    std::println("\n=== ALL PROPERTIES HELD ===");
    return 0;
}
