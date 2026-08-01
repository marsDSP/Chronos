// tests/harnesses/utils/bump_arena_check.cpp
// ──────────────────────────────────────────────────────────────────────────
// BumpArena correctness (C9). Gates the layout/instrumentation vehicle that
// backs the engine's scratch spans:
//
//  (1) ALIGNMENT — every pointer returned by allocate_bytes honors the
//      requested power-of-two alignment (4/8/16/32/64), including after odd-
//      sized carves that misalign the cursor.
//  (2) EXHAUSTION — a carve past the remaining bytes returns nullptr (the
//      function is noexcept, so a throw would terminate the process); the
//      arena stays usable for carves that still fit.
//  (3) clear() REWIND — after clear(), the identical carve sequence re-serves
//      the identical addresses in the identical order.
//  (4) BYTE ACCOUNTING — get_bytes_used() matches a hand-computed layout
//      (align_up of the cursor + n per carve); get_total_num_bytes() matches
//      the reset size.
//  (5) MOVE — move-construct-then-destroy-source leaves the moved-to arena's
//      storage valid (pattern written before the move reads back after the
//      source's destruction). This is the regression test for the reference-
//      member defect documented in the header: a reference cursor would bind
//      to the moved-from object's storage and dangle. Move-assign likewise.
//  (6) EDGES — default-constructed arena serves nullptr; allocate<float>
//      honors alignof and count; base region is kBaseAlignment-aligned.
//
// The CHRONOS_ARENA_DEBUG poison path in clear() is compile-time and trivial
// (a memset); it is not exercised here — the harness builds the production
// configuration.
//
// Conventions (matching latency_null_check / chain_parity): plain main(),
// exit code, printf, always-live CHECK/FAIL. Links SharedCode only; no JUCE.
// ──────────────────────────────────────────────────────────────────────────

#include "utils/memory/BumpArena.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace {

using MarsDSP::Memory::BumpArena;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

std::uintptr_t addr(const void* p) { return reinterpret_cast<std::uintptr_t>(p); }

// ── (1) alignment ─────────────────────────────────────────────────────────
void testAlignment()
{
    g_section = "alignment";
    BumpArena a;
    a.reset(4096);

    const std::size_t aligns[] = { 4, 8, 16, 32, 64 };
    const std::size_t sizes[]  = { 1, 3, 7, 13, 16, 24, 100 };
    for (std::size_t al : aligns)
        for (std::size_t n : sizes)
        {
            void* p = a.allocate_bytes(n, al);
            CHECK(p != nullptr);
            CHECK(addr(p) % al == 0);
        }
    std::printf("alignment (odd sizes, aligns 4..64): PASS\n");
}

// ── (2) exhaustion ────────────────────────────────────────────────────────
void testExhaustion()
{
    g_section = "exhaustion";
    BumpArena a;
    a.reset(256);

    void* p = a.allocate_bytes(200, 8);
    CHECK(p != nullptr);
    CHECK(a.get_bytes_used() == 200);

    // 100 bytes do not fit in the remaining 56: nullptr, cursor unmoved.
    CHECK(a.allocate_bytes(100, 8) == nullptr);
    CHECK(a.get_bytes_used() == 200);

    // 56 bytes fit exactly.
    void* q = a.allocate_bytes(56, 8);
    CHECK(q != nullptr);
    CHECK(a.get_bytes_used() == 256);

    // Full: even 1 byte fails. Alignment padding can also exhaust: a 64-
    // aligned 1-byte carve from a full arena has no room to pad into.
    CHECK(a.allocate_bytes(1, 8) == nullptr);
    CHECK(a.allocate_bytes(1, 64) == nullptr);
    CHECK(a.get_bytes_used() == 256);
    std::printf("exhaustion (nullptr, no throw, cursor unmoved): PASS\n");
}

// ── (3) clear() rewind re-serves identical addresses ──────────────────────
void testClearRewind()
{
    g_section = "clear-rewind";
    BumpArena a;
    a.reset(1024);

    constexpr int kSeq = 6;
    const std::size_t sizes[kSeq]  = { 13, 40, 1, 128, 7, 200 };
    const std::size_t aligns[kSeq] = { 8, 16, 4, 64, 8, 32 };
    void* first[kSeq] = {};

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
        void* p = a.allocate_bytes(sizes[i], aligns[i]);
        CHECK(p == first[i]);
    }
    std::printf("clear() rewind re-serves identical addresses: PASS\n");
}

// ── (4) byte accounting matches hand-computed layout ──────────────────────
void testByteAccounting()
{
    g_section = "byte-accounting";
    BumpArena a;
    a.reset(1024);
    CHECK(a.get_bytes_used() == 0);
    CHECK(a.get_total_num_bytes() == 1024);

    // Hand-computed: used <- align_up(used, align) + n.
    void* p = a.allocate_bytes(13, 8);    // align_up(0,8)=0   -> used 13
    CHECK(p != nullptr);
    CHECK(a.get_bytes_used() == 13);
    p = a.allocate_bytes(20, 16);         // align_up(13,16)=16 -> used 36
    CHECK(p != nullptr);
    CHECK(a.get_bytes_used() == 36);
    p = a.allocate_bytes(1, 64);          // align_up(36,64)=64 -> used 65
    CHECK(p != nullptr);
    CHECK(a.get_bytes_used() == 65);
    p = a.allocate_bytes(0, 8);           // zero-size carve: pads, adds nothing
    CHECK(p != nullptr);
    CHECK(a.get_bytes_used() == 72);

    std::printf("byte accounting (hand-computed layout): PASS\n");
}

// ── (5) move: construct-then-destroy-source, then move-assign ─────────────
void testMove()
{
    g_section = "move";
    void* saved = nullptr;
    constexpr std::size_t kBytes = 512;

    BumpArena b;
    {
        BumpArena a;
        a.reset(kBytes);
        saved = a.allocate_bytes(64, 16);
        CHECK(saved != nullptr);
        std::memset(saved, 0xAB, 64);           // pattern into the storage

        b = std::move(a);                       // move-ASSIGN; source empties
        CHECK(a.get_bytes_used() == 0);
        CHECK(a.get_total_num_bytes() == 0);
        CHECK(a.allocate_bytes(8, 8) == nullptr);
    }                                           // source destroyed here

    // The moved-to arena still owns the (intact) storage.
    CHECK(b.get_total_num_bytes() == kBytes);
    CHECK(static_cast<const unsigned char*>(saved)[0] == 0xAB);
    CHECK(static_cast<const unsigned char*>(saved)[63] == 0xAB);

    // Move-CONSTRUCT, then destroy the source.
    BumpArena d;
    {
        BumpArena c = std::move(b);             // move-CONSTRUCT
        CHECK(b.get_total_num_bytes() == 0);
        CHECK(c.get_total_num_bytes() == kBytes);
        CHECK(static_cast<const unsigned char*>(saved)[32] == 0xAB);
        d = std::move(c);
    }
    CHECK(d.get_total_num_bytes() == kBytes);
    CHECK(static_cast<const unsigned char*>(saved)[32] == 0xAB);
    std::printf("move construct/assign + destroy-source (no dangle): PASS\n");
}

// ── (6) edges: default-constructed, typed allocate, base alignment ────────
void testEdges()
{
    g_section = "edges";
    BumpArena a;                                // never reset
    CHECK(a.get_bytes_used() == 0);
    CHECK(a.get_total_num_bytes() == 0);
    CHECK(a.allocate_bytes(1, 8) == nullptr);
    CHECK(a.allocate<float>(4) == nullptr);

    BumpArena b;
    b.reset(256);
    float* f = b.allocate<float>(16);           // 64 bytes
    CHECK(f != nullptr);
    CHECK(addr(f) % alignof(float) == 0);
    CHECK(b.get_bytes_used() == 64);
    for (int i = 0; i < 16; ++i) f[i] = static_cast<float>(i);
    CHECK(f[15] == 15.0f);

    // base region itself is kBaseAlignment-aligned.
    void* base = b.allocate_bytes(1, BumpArena::kBaseAlignment);
    CHECK(addr(base) % BumpArena::kBaseAlignment == 0);

    // reset() re-allocates (prepare-time): old extents forgotten.
    b.reset(128);
    CHECK(b.get_bytes_used() == 0);
    CHECK(b.get_total_num_bytes() == 128);
    std::printf("edges (default arena, typed carve, base alignment): PASS\n");
}

} // namespace

int main()
{
    std::printf("=== Chronos bump_arena_check (C9) ===\n\n");
    testAlignment();
    testExhaustion();
    testClearRewind();
    testByteAccounting();
    testMove();
    testEdges();
    std::printf("\n=== ALL PROPERTIES HELD ===\n");
    return 0;
}
