// tests/harnesses/cd/ring_buffer_check.cpp
// ──────────────────────────────────────────────────────────────────────────
// Correctness harness for MarsDSP::Delays::Pow2RingBuffer, the single-channel
// storage layer under SimdDelayLine. Validates the write/mirror/read geometry
// against a naive modulo oracle (pure %, no mirror, no memcpy — obviously
// correct by inspection).
//
//   1. Zero state      – after prepare(), [0, capacity + kTail) is zero.
//   2. Block write     – for every start in [0, capacity) and every length in
//                        [1, maxBlock], writeBlock matches the model and the
//                        mirror invariant holds after refreshMirror.
//   3. Window read     – ramp-filled, every start in [0, capacity) × every
//                        length in [1, kSubBlock + kTail], readWindow == oracle.
//   4. Interleaved     – a few thousand pseudo-random blocks driven through
//                        buffer + model together, reading windows after each.
//                        Catches refreshMirror/write ordering bugs.
//   5. Large capacity  – sampled subset at 1 << 18 with starts clustered near
//                        the wrap, catching overflow in startIdx + length.
//   6. windowPtr parity – for every (startIdx, len) where windowPtr is
//                        non-null, it returns the same data as readWindow;
//                        and windowPtr returns null exactly when a naive
//                        modulo oracle says the window wraps past the mirror.
//
// Conventions (matching tan_bench): plain main(), exit code, printf, always-
// live CHECK/FAIL (NOT assert — NDEBUG in Release would void every test).
// Links SharedCode only; no JUCE.
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/Pow2RingBuffer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kTail      = MarsDSP::Delays::Pow2RingBuffer::kTail; // 8
constexpr int kSubBlock  = 16;
constexpr int kMaxWindow = kSubBlock + kTail;   // 24
constexpr int kTestCap   = 64;                  // pow2, comfortably above kTail & kMaxWindow
constexpr int kMaxBlock  = kTestCap / 2;        // 32
constexpr int kLargeCap  = 1 << 18;             // 262144, matches the reference kBufSize

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

// Naive modulo oracle: pure %, no mirror, no memcpy. Obviously correct.
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

    // ── 1. Zero state ──────────────────────────────────────────────────────
    g_section = "zero state";
    {
        Pow2RingBuffer buf;
        buf.prepare(kTestCap);
        const int cap = buf.getCapacity();
        CHECK(cap == kTestCap);
        CHECK(buf.mask() == cap - 1);

        // Read the whole canonical region + mirror via readWindow (start=0,
        // len=cap) and then a window that peeks at the mirror.
        std::vector<float> got(static_cast<std::size_t>(cap));
        buf.readWindow(got.data(), 0, cap);
        for (int i = 0; i < cap; ++i)
            if (got[i] != 0.0f) FAIL("canonical i=%d is %g (expected 0)", i, (double)got[i]);

        // Peek at the mirror: read a window starting at cap-1 of length kTail+1.
        // The first sample is canonical[cap-1]=0, the next kTail are mirror[0..kTail).
        std::vector<float> mir(kTail + 1);
        buf.readWindow(mir.data(), cap - 1, kTail + 1);
        for (int k = 0; k <= kTail; ++k)
            if (mir[k] != 0.0f) FAIL("mirror peek k=%d is %g (expected 0)", k, (double)mir[k]);

        std::printf("zero state: PASS\n");
    }

    // ── 2. Block write + mirror invariant ──────────────────────────────────
    g_section = "block write + mirror";
    {
        Pow2RingBuffer buf;
        buf.prepare(kTestCap);
        const int cap = buf.getCapacity();
        oracle.init(cap);

        std::vector<float> blk, got;
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

                // Canonical region matches the model.
                got.resize(static_cast<std::size_t>(cap));
                buf.readWindow(got.data(), 0, cap);
                for (int i = 0; i < cap; ++i)
                    if (got[i] != oracle.model[i])
                        FAIL("start=%d n=%d canonical i=%d got=%g exp=%g",
                             start, n, i, (double)got[i], (double)oracle.model[i]);

                // Mirror invariant: readWindow at cap-1 of length kTail+1 gives
                // canonical[cap-1] followed by mirror[0..kTail-1], which must
                // equal canonical[0..kTail-1] (i.e. oracle.model[0..kTail-1]).
                std::vector<float> peek(kTail + 1);
                buf.readWindow(peek.data(), cap - 1, kTail + 1);
                if (peek[0] != oracle.model[cap - 1])
                    FAIL("start=%d n=%d mirror peek[0] got=%g exp=%g",
                         start, n, (double)peek[0], (double)oracle.model[cap - 1]);
                for (int k = 0; k < kTail; ++k)
                    if (peek[k + 1] != oracle.model[k])
                        FAIL("start=%d n=%d mirror k=%d got=%g exp=%g",
                             start, n, k, (double)peek[k + 1], (double)oracle.model[k]);
            }
        }
        std::printf("block write + mirror invariant: PASS\n");
    }

    // ── 3. Window read (exhaustive) ────────────────────────────────────────
    g_section = "window read";
    {
        Pow2RingBuffer buf;
        buf.prepare(kTestCap);
        const int cap = buf.getCapacity();
        oracle.init(cap);

        // Fill with a unique ramp so every logical index carries a distinct
        // value; any duplication/off-by-one in readWindow shows up.
        std::vector<float> ramp(static_cast<std::size_t>(cap));
        for (int i = 0; i < cap; ++i)
            ramp[i] = static_cast<float>(i + 1); // 1-based so 0 is distinguishable from uninitialised
        buf.writeBlock(ramp.data(), 0, cap);
        buf.refreshMirror(0, cap);
        oracle.write(0, ramp.data(), cap);

        std::vector<float> got, exp;
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
                        FAIL("start=%d len=%d j=%d got=%g exp=%g",
                             start, len, j, (double)got[j], (double)exp[j]);
            }
        }
        std::printf("window read [1,%d]: PASS\n", kMaxWindow);
    }

    // ── 4. Interleaved sequence ────────────────────────────────────────────
    g_section = "interleaved";
    {
        Pow2RingBuffer buf;
        buf.prepare(kTestCap);
        const int cap = buf.getCapacity();
        oracle.init(cap);

        Rng rng(20240725u);
        constexpr int kBlocks = 4000;
        int writeIdx = 0; // caller owns the write index
        std::vector<float> blk, got, exp;
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

            // Read a spread of windows: the just-written block, a random
            // start, and a start at the wrap neighbourhood.
            const int readStarts[3] = {
                (writeIdx - n + cap) & buf.mask(), // block start (pre-advance)
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
                            FAIL("block %d rs=%d len=%d j=%d got=%g exp=%g",
                                 b, rs, len, j, (double)got[j], (double)exp[j]);
                }
            }
        }
        std::printf("interleaved sequence (%d blocks): PASS\n", kBlocks);
    }

    // ── 5. Large capacity (1 << 18) ────────────────────────────────────────
    g_section = "large capacity";
    {
        Pow2RingBuffer buf;
        buf.prepare(kLargeCap);
        const int cap = buf.getCapacity();
        CHECK(cap == kLargeCap);
        oracle.init(cap);

        // Fill with a unique ramp via a single full-capacity writeBlock.
        std::vector<float> ramp(static_cast<std::size_t>(cap));
        for (int i = 0; i < cap; ++i)
            ramp[i] = static_cast<float>(i + 1);
        buf.writeBlock(ramp.data(), 0, cap);
        buf.refreshMirror(0, cap);
        oracle.write(0, ramp.data(), cap);

        // Starts clustered near the wrap (last 64 indices), every legal window
        // length. This is where an overflow in startIdx + length or a mirror
        // sized too small would surface at realistic magnitude.
        std::vector<float> got, exp;
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
                        FAIL("start=%d len=%d j=%d got=%g exp=%g",
                             start, len, j, (double)got[j], (double)exp[j]);
            }
        }

        // Mirror invariant at scale.
        std::vector<float> peek(kTail + 1);
        buf.readWindow(peek.data(), cap - 1, kTail + 1);
        if (peek[0] != oracle.model[cap - 1])
            FAIL("large-cap mirror peek[0] got=%g exp=%g",
                 (double)peek[0], (double)oracle.model[cap - 1]);
        for (int k = 0; k < kTail; ++k)
            if (peek[k + 1] != oracle.model[k])
                FAIL("large-cap mirror k=%d got=%g exp=%g",
                     k, (double)peek[k + 1], (double)oracle.model[k]);

        std::printf("large capacity (1<<18), wrap-clustered: PASS\n");
    }

    // ── 6. windowPtr parity (C6) ──────────────────────────────────────────
    g_section = "windowPtr parity";
    {
        Pow2RingBuffer buf;
        buf.prepare(kTestCap);
        const int cap = buf.getCapacity();
        oracle.init(cap);

        // Ramp-fill so every index carries a distinct value.
        std::vector<float> ramp(static_cast<std::size_t>(cap));
        for (int i = 0; i < cap; ++i) ramp[i] = static_cast<float>(i + 1);
        buf.writeBlock(ramp.data(), 0, cap);
        buf.refreshMirror(0, cap);
        oracle.write(0, ramp.data(), cap);

        std::vector<float> viaRead, viaPtr;
        for (int start = 0; start < cap; ++start)
        {
            for (int len = 1; len <= kMaxWindow; ++len)
            {
                viaRead.resize(static_cast<std::size_t>(len));
                viaPtr.resize(static_cast<std::size_t>(len));
                buf.readWindow(viaRead.data(), start, len);
                const float* p = buf.windowPtr(start, len);

                // Naive modulo oracle: does the window wrap past the mirror?
                const bool wraps = (start + len) > (cap + kTail);

                if (p == nullptr)
                {
                    if (!wraps)
                        FAIL("start=%d len=%d: windowPtr null but oracle says contiguous", start, len);
                }
                else
                {
                    if (wraps)
                        FAIL("start=%d len=%d: windowPtr non-null but oracle says wrap", start, len);
                    for (int j = 0; j < len; ++j)
                        viaPtr[static_cast<std::size_t>(j)] = p[j];
                    for (int j = 0; j < len; ++j)
                        if (viaPtr[static_cast<std::size_t>(j)] != viaRead[static_cast<std::size_t>(j)])
                            FAIL("start=%d len=%d j=%d: ptr=%g read=%g",
                                 start, len, j,
                                 (double)viaPtr[static_cast<std::size_t>(j)],
                                 (double)viaRead[static_cast<std::size_t>(j)]);
                }
            }
        }
        std::printf("windowPtr parity (contiguous == readWindow, null == wrap oracle): PASS\n");
    }

    return 0;
}

} // namespace

int main()
{
    std::printf("=== Chronos Pow2RingBuffer correctness harness ===\n");
    std::printf("kTail=%d  kMaxWindow=%d  testCapacity=%d  largeCapacity=%d\n\n",
                kTail, kMaxWindow, kTestCap, kLargeCap);

    int r = runAll();

    std::printf("\n=== %s ===\n", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}
