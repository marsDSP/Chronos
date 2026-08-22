/**
 * Correctness harness for Pow2RingBuffer, the single-channel storage layer
 * under SimdDelayLine. Validates write, mirror, and read geometry against a
 * naive modulo oracle. Plain main(), exit code, always-live CHECK/FAIL.
 */

#include "dsp/Pow2RingBuffer.h"
#include "utils/memory/BumpArena.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <print>
#include <vector>

namespace {

constexpr int kTail      = MarsDSP::Delays::Pow2RingBuffer::kTail;
constexpr int kSubBlock  = 16;
constexpr int kMaxWindow = kSubBlock + kTail;
constexpr int kTestCap   = 64;
constexpr int kMaxBlock  = kTestCap / 2;
constexpr int kLargeCap  = 1 << 18;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

/// Naive modulo oracle: pure %, no mirror, no memcpy.
struct Oracle
{
    std::vector<float> model;
    int capacity = 0;

    void init(int cap)
    {
        capacity = cap;
        model.assign(static_cast<std::size_t>(cap), 0.0f);
    }
    void write(int start, const float* src, int n)
    {
        for (int k = 0; k < n; ++k)
            model[static_cast<std::size_t>((start + k) % capacity)] = src[k];
    }
    void read(int start, float* dst, int len) const
    {
        for (int j = 0; j < len; ++j)
            dst[j] = model[static_cast<std::size_t>((start + j) % capacity)];
    }
};

struct Rng
{
    std::uint32_t s;
    explicit Rng(std::uint32_t seed) : s(seed) {}
    std::uint32_t next() { s = s * 1664525u + 1013904223u; return s; }
    int range(int lo, int hi) { return lo + static_cast<int>(next() % static_cast<std::uint32_t>(hi - lo + 1)); }
};

int runAll()
{
    using MarsDSP::Delays::Pow2RingBuffer;
    Oracle oracle;

    // 1. Zero state.
    g_section = "zero state";
    {
        Pow2RingBuffer buf;
        buf.prepare(kTestCap);
        const int cap = buf.getCapacity();
        CHECK(cap == kTestCap);
        CHECK(buf.mask() == cap - 1);

        std::vector<float> got(static_cast<std::size_t>(cap));
        buf.readWindow(got.data(), 0, cap);
        for (int i = 0; i < cap; ++i)
            if (got[i] != 0.0f) FAIL("canonical i={{}} is {{}} (expected 0)", i, static_cast<double>(got[i]));

        // Peek at the mirror: read a window starting at cap-1 of length kTail+1.
        std::vector<float> mir(kTail + 1);
        buf.readWindow(mir.data(), cap - 1, kTail + 1);
        for (int k = 0; k <= kTail; ++k)
            if (mir[k] != 0.0f) FAIL("mirror peek k={{}} is {{}} (expected 0)", k, static_cast<double>(mir[k]));

        std::println("zero state: PASS");
    }

    // 2. Block write and mirror invariant.
    g_section = "block write + mirror";
    {
        Pow2RingBuffer buf;
        buf.prepare(kTestCap);
        const int cap = buf.getCapacity();
        oracle.init(cap);

        std::vector<float> blk;
        std::vector<float> got;
        for (int start = 0; start < cap; ++start)
        {
            for (int n = 1; n <= kMaxBlock; ++n)
            {
                blk.resize(static_cast<std::size_t>(n));
                for (int k = 0; k < n; ++k)
                    blk[k] = static_cast<float>(1 + start * 1000 + n * 10 + k);

                buf.writeBlock(blk.data(), start, n);
                buf.refreshMirror(start, n);
                oracle.write(start, blk.data(), n);

                got.resize(static_cast<std::size_t>(cap));
                buf.readWindow(got.data(), 0, cap);
                for (int i = 0; i < cap; ++i)
                    if (got[i] != oracle.model[i])
                        FAIL("start={{}} n={{}} canonical i={{}} got={{}} exp={{}}",
                             start, n, i, static_cast<double>(got[i]), static_cast<double>(oracle.model[i]));

                // Mirror invariant: readWindow at cap-1 of length kTail+1 gives
                // canonical[cap-1] followed by mirror[0..kTail-1], which must
                // equal canonical[0..kTail-1].
                std::vector<float> peek(kTail + 1);
                buf.readWindow(peek.data(), cap - 1, kTail + 1);
                if (peek[0] != oracle.model[cap - 1])
                    FAIL("start={{}} n={{}} mirror peek[0] got={{}} exp={{}}",
                         start, n, static_cast<double>(peek[0]), static_cast<double>(oracle.model[cap - 1]));
                for (int k = 0; k < kTail; ++k)
                    if (peek[k + 1] != oracle.model[k])
                        FAIL("start={{}} n={{}} mirror k={{}} got={{}} exp={{}}",
                             start, n, k, static_cast<double>(peek[k + 1]), static_cast<double>(oracle.model[k]));
            }
        }
        std::println("block write + mirror invariant: PASS");
    }

    // 3. Window read (exhaustive).
    g_section = "window read";
    {
        Pow2RingBuffer buf;
        buf.prepare(kTestCap);
        const int cap = buf.getCapacity();
        oracle.init(cap);

        // Fill with a unique ramp so every logical index carries a distinct value.
        std::vector<float> ramp(static_cast<std::size_t>(cap));
        for (int i = 0; i < cap; ++i)
            ramp[i] = static_cast<float>(i + 1);
        buf.writeBlock(ramp.data(), 0, cap);
        buf.refreshMirror(0, cap);
        oracle.write(0, ramp.data(), cap);

        std::vector<float> got;
        std::vector<float> exp;
        for (int start = 0; start < cap; ++start)
        {
            for (int len = 1; len <= kMaxWindow; ++len)
            {
                got.resize(static_cast<std::size_t>(len));
                exp.resize(static_cast<std::size_t>(len));
                buf.readWindow(got.data(), start, len);
                oracle.read(start, exp.data(), len);
                for (int j = 0; j < len; ++j)
                    if (got[j] != exp[j])
                        FAIL("start={{}} len={{}} j={{}} got={{}} exp={{}}",
                             start, len, j, static_cast<double>(got[j]), static_cast<double>(exp[j]));
            }
        }
        std::println("window read [1,{}]: PASS", kMaxWindow);
    }

    // 4. Interleaved sequence.
    g_section = "interleaved";
    {
        Pow2RingBuffer buf;
        buf.prepare(kTestCap);
        const int cap = buf.getCapacity();
        oracle.init(cap);

        Rng rng(20240725u);
        constexpr int kBlocks = 4000;
        int writeIdx = 0;
        std::vector<float> blk;
        std::vector<float> got;
        std::vector<float> exp;
        for (int b = 0; b < kBlocks; ++b)
        {
            const int n = rng.range(1, kMaxBlock);
            blk.resize(static_cast<std::size_t>(n));
            for (int k = 0; k < n; ++k)
                blk[k] = static_cast<float>(b * 10 + 1 + k);

            buf.writeBlock(blk.data(), writeIdx, n);
            buf.refreshMirror(writeIdx, n);
            oracle.write(writeIdx, blk.data(), n);
            writeIdx = (writeIdx + n) & buf.mask();

            const std::array<int, 3> readStarts = {
                (writeIdx - n + cap) & buf.mask(),
                rng.range(0, cap - 1),
                (writeIdx - kMaxWindow + cap) & buf.mask()
            };
            for (int rs : readStarts)
            {
                for (int len = 1; len <= kMaxWindow; ++len)
                {
                    got.resize(static_cast<std::size_t>(len));
                    exp.resize(static_cast<std::size_t>(len));
                    buf.readWindow(got.data(), rs, len);
                    oracle.read(rs, exp.data(), len);
                    for (int j = 0; j < len; ++j)
                        if (got[j] != exp[j])
                            FAIL("block {{}} rs={{}} len={{}} j={{}} got={{}} exp={{}}",
                                 b, rs, len, j, static_cast<double>(got[j]), static_cast<double>(exp[j]));
                }
            }
        }
        std::println("interleaved sequence ({} blocks): PASS", kBlocks);
    }

    // 5. Large capacity (1 << 18).
    g_section = "large capacity";
    {
        Pow2RingBuffer buf;
        buf.prepare(kLargeCap);
        const int cap = buf.getCapacity();
        CHECK(cap == kLargeCap);
        oracle.init(cap);

        std::vector<float> ramp(static_cast<std::size_t>(cap));
        for (int i = 0; i < cap; ++i)
            ramp[i] = static_cast<float>(i + 1);
        buf.writeBlock(ramp.data(), 0, cap);
        buf.refreshMirror(0, cap);
        oracle.write(0, ramp.data(), cap);

        // Starts clustered near the wrap, every legal window length.
        std::vector<float> got;
        std::vector<float> exp;
        const int clusterStart = cap - 64;
        for (int start = clusterStart; start < cap; ++start)
        {
            for (int len = 1; len <= kMaxWindow; ++len)
            {
                got.resize(static_cast<std::size_t>(len));
                exp.resize(static_cast<std::size_t>(len));
                buf.readWindow(got.data(), start, len);
                oracle.read(start, exp.data(), len);
                for (int j = 0; j < len; ++j)
                    if (got[j] != exp[j])
                        FAIL("start={{}} len={{}} j={{}} got={{}} exp={{}}",
                             start, len, j, static_cast<double>(got[j]), static_cast<double>(exp[j]));
            }
        }

        std::vector<float> peek(kTail + 1);
        buf.readWindow(peek.data(), cap - 1, kTail + 1);
        if (peek[0] != oracle.model[cap - 1])
            FAIL("large-cap mirror peek[0] got={{}} exp={{}}",
                 static_cast<double>(peek[0]), static_cast<double>(oracle.model[cap - 1]));
        for (int k = 0; k < kTail; ++k)
            if (peek[k + 1] != oracle.model[k])
                FAIL("large-cap mirror k={{}} got={{}} exp={{}}",
                     k, static_cast<double>(peek[k + 1]), static_cast<double>(oracle.model[k]));

        std::println("large capacity (1<<18), wrap-clustered: PASS");
    }

    // 6. windowPtr parity.
    g_section = "windowPtr parity";
    {
        Pow2RingBuffer buf;
        buf.prepare(kTestCap);
        const int cap = buf.getCapacity();
        oracle.init(cap);

        std::vector<float> ramp(static_cast<std::size_t>(cap));
        for (int i = 0; i < cap; ++i) ramp[i] = static_cast<float>(i + 1);
        buf.writeBlock(ramp.data(), 0, cap);
        buf.refreshMirror(0, cap);
        oracle.write(0, ramp.data(), cap);

        std::vector<float> viaRead;
        std::vector<float> viaPtr;
        for (int start = 0; start < cap; ++start)
        {
            for (int len = 1; len <= kMaxWindow; ++len)
            {
                viaRead.resize(static_cast<std::size_t>(len));
                viaPtr.resize(static_cast<std::size_t>(len));
                buf.readWindow(viaRead.data(), start, len);
                const float* p = buf.windowPtr(start, len);

                const bool wraps = (start + len) > (cap + kTail);

                if (p == nullptr)
                {
                    if (!wraps)
                        FAIL("start={{}} len={{}}: windowPtr null but oracle says contiguous", start, len);
                }
                else
                {
                    if (wraps)
                        FAIL("start={{}} len={{}}: windowPtr non-null but oracle says wrap", start, len);
                    for (int j = 0; j < len; ++j)
                        viaPtr[static_cast<std::size_t>(j)] = p[j];
                    for (int j = 0; j < len; ++j)
                        if (viaPtr[static_cast<std::size_t>(j)] != viaRead[static_cast<std::size_t>(j)])
                            FAIL("start={{}} len={{}} j={{}}: ptr={{}} read={{}}",
                                 start, len, j,
                                 static_cast<double>(viaPtr[static_cast<std::size_t>(j)]),
                                 static_cast<double>(viaRead[static_cast<std::size_t>(j)]));
                }
            }
        }
        std::println("windowPtr parity (contiguous == readWindow, null == wrap oracle): PASS");
    }

    // 7. Arena-backed storage parity.
    g_section = "arena-backed";
    {
        using MarsDSP::Memory::BumpArena;

        const std::size_t floatsOne = Pow2RingBuffer::arenaFloatsFor(kTestCap);
        CHECK(floatsOne >= static_cast<std::size_t>(kTestCap + kTail));
        CHECK(floatsOne % 16 == 0);

        BumpArena arena;
        arena.reset(2 * floatsOne * sizeof(float));

        Pow2RingBuffer buf;
        buf.prepare(kTestCap, arena);
        CHECK(buf.getCapacity() == kTestCap);
        CHECK(buf.mask() == kTestCap - 1);
        CHECK(arena.get_bytes_used() == floatsOne * sizeof(float));

        Pow2RingBuffer own;
        own.prepare(kTestCap);
        oracle.init(kTestCap);

        const float* base = buf.windowPtr(0, 1);
        CHECK(base != nullptr);
        CHECK(reinterpret_cast<std::uintptr_t>(base) % BumpArena::kBaseAlignment == 0);

        {
            std::vector<float> got(static_cast<std::size_t>(kTestCap));
            buf.readWindow(got.data(), 0, kTestCap);
            for (int i = 0; i < kTestCap; ++i)
                if (got[i] != 0.0f) FAIL("arena zero state i={{}} is {{}}", i, static_cast<double>(got[i]));
        }

        // Interleaved pseudo-random sequence through both rings and oracle.
        // The arena-backed ring must be bit-identical to the owning ring.
        Rng rng(20260801u);
        constexpr int kBlocks = 2000;
        int wA = 0;
        int wO = 0;
        std::vector<float> blk;
        std::vector<float> gotA;
        std::vector<float> gotO;
        std::vector<float> exp;
        for (int b = 0; b < kBlocks; ++b)
        {
            const int n = rng.range(1, kMaxBlock);
            blk.resize(static_cast<std::size_t>(n));
            for (int k = 0; k < n; ++k)
                blk[k] = static_cast<float>(b * 10 + 1 + k);

            buf.writeBlock(blk.data(), wA, n);
            buf.refreshMirror(wA, n);
            own.writeBlock(blk.data(), wO, n);
            own.refreshMirror(wO, n);
            oracle.write(wA, blk.data(), n);
            wA = (wA + n) & buf.mask();
            wO = (wO + n) & own.mask();

            const std::array<int, 3> readStarts = {
                (wA - n + kTestCap) & buf.mask(),
                rng.range(0, kTestCap - 1),
                (wA - kMaxWindow + kTestCap) & buf.mask()
            };
            for (int rs : readStarts)
            {
                for (int len = 1; len <= kMaxWindow; ++len)
                {
                    gotA.resize(static_cast<std::size_t>(len));
                    gotO.resize(static_cast<std::size_t>(len));
                    exp.resize(static_cast<std::size_t>(len));
                    buf.readWindow(gotA.data(), rs, len);
                    own.readWindow(gotO.data(), rs, len);
                    oracle.read(rs, exp.data(), len);
                    for (int j = 0; j < len; ++j)
                    {
                        if (gotA[static_cast<std::size_t>(j)] != exp[static_cast<std::size_t>(j)])
                            FAIL("arena block {{}} rs={{}} len={{}} j={{}} got={{}} exp={{}}",
                                 b, rs, len, j, static_cast<double>(gotA[static_cast<std::size_t>(j)]),
                                 static_cast<double>(exp[static_cast<std::size_t>(j)]));
                        if (gotA[static_cast<std::size_t>(j)] != gotO[static_cast<std::size_t>(j)])
                            FAIL("arena vs owning block {{}} rs={{}} len={{}} j={{}} {{}} != {{}}",
                                 b, rs, len, j, static_cast<double>(gotA[static_cast<std::size_t>(j)]),
                                 static_cast<double>(gotO[static_cast<std::size_t>(j)]));
                    }
                    const float* p = buf.windowPtr(rs, len);
                    const bool wraps = (rs + len) > (kTestCap + kTail);
                    if (p == nullptr) { CHECK(wraps); }
                    else
                    {
                        CHECK(!wraps);
                        for (int j = 0; j < len; ++j)
                            CHECK(p[j] == exp[static_cast<std::size_t>(j)]);
                    }
                }
            }
        }

        // A second ring carved from the same arena lands immediately after the first.
        Pow2RingBuffer buf2;
        buf2.prepare(kTestCap, arena);
        CHECK(arena.get_bytes_used() == 2 * floatsOne * sizeof(float));
        const float* base2 = buf2.windowPtr(0, 1);
        CHECK(base2 == base + floatsOne);
        CHECK(reinterpret_cast<std::uintptr_t>(base2) % BumpArena::kBaseAlignment == 0);

        std::println("arena-backed storage (aligned, exact accounting, owning parity): PASS");
    }

    return 0;
}

} // namespace

int main()
{
    std::println("=== Chronos Pow2RingBuffer correctness harness ===");
    std::println("kTail={}  kMaxWindow={}  testCapacity={}  largeCapacity={}",
                kTail, kMaxWindow, kTestCap, kLargeCap);
    std::println();

    const int r = runAll();

    std::println();
    std::println("=== {} ===", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}
