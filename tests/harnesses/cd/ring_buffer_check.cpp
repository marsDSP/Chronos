// tests/harnesses/cd/ring_buffer_check.cpp
// ──────────────────────────────────────────────────────────────────────────
// Correctness harness for MarsDSP::Buffers::Pow2RingBuffer, the storage layer
// under the future SIMD delay engine. Validates every geometric invariant the
// engine will rely on against a naive modulo oracle (pure %, no mirror, no
// memcpy, no cleverness — obviously correct by inspection).
//
//   1. Zero state        – after prepare(), [0, capacity + kMirrorSamples) is
//                          zero on every channel and getWriteIndex() == 0.
//   2. Block write       – for every start in [0, capacity) and every length in
//                          [1, maxBlock], writeAt matches the model index-for-
//                          index in the canonical region AND the mirror
//                          invariant storage[cap + k] == storage[k] holds.
//   3. Window read       – ramp-filled, every start × every length in
//                          [1, kMaxWindowSamples], readWindow == modelRead.
//   4. Contiguity        – over the same matrix, tryGetContiguous never returns
//                          null (mirror >= kMaxWindow-1) and points at the model.
//   5. Interleaved       – a few thousand pseudo-random blocks driven through
//                          buffer + model together, reading a spread of windows
//                          after each; also asserts the write index tracks the
//                          model's. Catches refresh/advance ordering bugs that
//                          per-operation tests cannot.
//   6. Large capacity    – a sampled subset at capacity = 1<<18 with starts
//                          clustered near the wrap, catching anything that only
//                          manifests at realistic magnitudes (e.g. overflow in
//                          startIndex + length).
//
// Conventions (matching tests/harnesses/perf/tan_bench.cpp):
//   - Plain main(), exit code (0 = every property held), printf reports.
//   - Locally-defined always-live CHECK/FAIL (NOT assert): NDEBUG in a Release
//     configure would silently void every assert-based test, so the test logic
//     must not depend on it. The header's own assert preconditions stay armed
//     in a Debug configure because this target is NOT forced to -O2.
//   - Links SharedCode only (source/ include path + cxx_std_23); no JUCE.
//
// Build:  cmake -S . -B build -DBUILD_TEST_HARNESSES=ON
//         cmake --build build --target ring_buffer_check
// Run:    ./build/tests/ring_buffer_check
// Exit:   0 = every property held, non-zero = a property failed.
//
// Sanitizer pass: uncomment the -fsanitize=address,undefined lines in
// tests/CMakeLists.txt, rebuild, and run once. The two-memcpy split and the
// mirror region are exactly where an off-by-one reads/writes one element past
// the allocation; ASan catches that class of bug far more directly than a value
// comparison does.
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/Pow2RingBuffer.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// kMirrorSamples matches the class default and the engine's window geometry:
// kSubBlock + kTail = 16 + 8 = 24 = kMaxWindowSamples, and the contiguity
// property needs mirror >= kMaxWindow - 1 = 23, which 32 satisfies.
constexpr int kMirror       = 32;
constexpr int kSubBlock     = 16;
constexpr int kTail         = 8;
constexpr int kMaxWindow    = kSubBlock + kTail;   // 24
constexpr int kTestCapacity = 64;                  // pow2, comfortably above mirror & window
constexpr int kMaxBlock     = kTestCapacity / 2;   // 32: blocks can wrap
constexpr int kLargeCap     = 1 << 18;             // 262144, matches the reference kBufSize

// Always-live test predicates. g_section gives every failure a locatable tag.
const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

// Naive modulo oracle. Pure %, no mirror, no memcpy. Obviously correct.
// Holds one model per channel so it mirrors the buffer's independent
// per-channel storage (a single shared model would conflate the two channels
// and false-fail as soon as channel 1 overwrites channel 0's indices).
template <typename T>
struct Oracle
{
    std::array<std::vector<T>, 2> model {};
    int capacity = 0;
    int writeIdx = 0;   // shared: both channels are pushed together

    void init(int cap)
    {
        capacity = cap;
        for (auto& m : model)
            m.assign(static_cast<std::size_t>(cap), T(0));
        writeIdx = 0;
    }
    void write(int ch, int start, const T* src, int n)
    {
        for (int k = 0; k < n; ++k)
            model[ch][static_cast<std::size_t>((start + k) % capacity)] = src[k];
    }
    void read(int ch, int start, T* dst, int len) const
    {
        for (int j = 0; j < len; ++j)
            dst[j] = model[ch][static_cast<std::size_t>((start + j) % capacity)];
    }
    void advance(int n) { writeIdx = (writeIdx + n) % capacity; }
};

// Read the full canonical region [0, capacity) into dst. Contiguous (start=0,
// len=capacity <= capacity + mirror), so readWindow takes the fast path.
template <typename T>
void readCanonical(const MarsDSP::Buffers::Pow2RingBuffer<T, kMirror>& buf, int ch, T* dst)
{
    buf.readWindow(ch, 0, dst, buf.getCapacity());
}

// Pointer to the canonical head [0, kMirror), via the contiguous fast path.
template <typename T>
const T* headPtr(const MarsDSP::Buffers::Pow2RingBuffer<T, kMirror>& buf, int ch)
{
    return buf.tryGetContiguous(ch, 0, kMirror);
}

// Pointer to the mirror region [capacity, capacity + kMirror). Reading a window
// that starts at capacity-1 and overhangs by kMirror lands entirely in the
// mirror; tryGetContiguous returns base + (capacity-1), so +1 is the mirror.
template <typename T>
const T* mirrorPtr(const MarsDSP::Buffers::Pow2RingBuffer<T, kMirror>& buf, int ch)
{
    const int cap = buf.getCapacity();
    const T* p = buf.tryGetContiguous(ch, cap - 1, kMirror + 1);
    return p ? p + 1 : nullptr;
}

// Reproducible LCG for the interleaved sequence.
struct Rng
{
    std::uint32_t s;
    explicit Rng(std::uint32_t seed) : s(seed) {}
    std::uint32_t next() { s = s * 1664525u + 1013904223u; return s; }
    int range(int lo, int hi) { return lo + static_cast<int>(next() % static_cast<std::uint32_t>(hi - lo + 1)); }
};

template <typename T>
int runAll(const char* name)
{
    using Buf = MarsDSP::Buffers::Pow2RingBuffer<T, kMirror>;
    Oracle<T> oracle;

    // ── 1. Zero state ──────────────────────────────────────────────────────
    g_section = "zero state";
    {
        Buf buf;
        buf.prepare(2, kTestCapacity);
        const int cap = buf.getCapacity();
        CHECK(cap == kTestCapacity);
        CHECK(buf.getMask() == cap - 1);
        CHECK(buf.getNumChannels() == 2);
        CHECK(buf.getWriteIndex() == 0);

        std::vector<T> got(static_cast<std::size_t>(cap));
        for (int ch = 0; ch < 2; ++ch)
        {
            readCanonical(buf, ch, got.data());
            for (int i = 0; i < cap; ++i)
                if (got[i] != T(0)) FAIL("canonical ch=%d i=%d is %g (expected 0)", ch, i, (double)got[i]);

            const T* mir = mirrorPtr(buf, ch);
            CHECK(mir != nullptr);
            for (int k = 0; k < kMirror; ++k)
                if (mir[k] != T(0)) FAIL("mirror ch=%d k=%d is %g (expected 0)", ch, k, (double)mir[k]);
        }
        std::printf("[%s] zero state: PASS\n", name);
    }

    // ── 2. Block write + mirror invariant ──────────────────────────────────
    g_section = "block write + mirror";
    {
        Buf buf;
        buf.prepare(2, kTestCapacity);
        const int cap = buf.getCapacity();
        oracle.init(cap);

        std::vector<T> blk, got;
        for (int ch = 0; ch < 2; ++ch)
        {
            for (int start = 0; start < cap; ++start)
            {
                for (int n = 1; n <= kMaxBlock; ++n)
                {
                    // Distinct ramp encoding (ch, start, n) so a misplacement
                    // or a stale mirror shows up as a value mismatch.
                    blk.resize(static_cast<std::size_t>(n));
                    for (int k = 0; k < n; ++k)
                        blk[k] = static_cast<T>(1 + ch * 1'000'000 + start * 1000 + n * 10 + k);

                    buf.writeAt(ch, start, blk.data(), n);
                    buf.refreshMirror();
                    oracle.write(ch, start, blk.data(), n);

                    // Canonical region matches the model index-for-index.
                    got.resize(static_cast<std::size_t>(cap));
                    readCanonical(buf, ch, got.data());
                    for (int i = 0; i < cap; ++i)
                        if (got[i] != oracle.model[ch][i])
                            FAIL("ch=%d start=%d n=%d canonical i=%d got=%g exp=%g",
                                 ch, start, n, i, (double)got[i], (double)oracle.model[ch][i]);

                    // Mirror invariant: storage[cap + k] == storage[k].
                    const T* head = headPtr(buf, ch);
                    const T* mir  = mirrorPtr(buf, ch);
                    CHECK(head != nullptr && mir != nullptr);
                    for (int k = 0; k < kMirror; ++k)
                        if (mir[k] != head[k])
                            FAIL("ch=%d start=%d n=%d mirror k=%d mir=%g head=%g",
                                 ch, start, n, k, (double)mir[k], (double)head[k]);
                }
            }
        }
        std::printf("[%s] block write + mirror invariant: PASS\n", name);
    }

    // ── 3. Window read ─────────────────────────────────────────────────────
    g_section = "window read";
    {
        Buf buf;
        buf.prepare(2, kTestCapacity);
        const int cap = buf.getCapacity();
        oracle.init(cap);

        // Fill with a unique ramp (index + channel offset) so every logical
        // index carries a distinct value; any duplication/off-by-one in
        // readWindow shows up as a mismatch.
        std::vector<T> ramp(static_cast<std::size_t>(cap));
        for (int ch = 0; ch < 2; ++ch)
        {
            for (int i = 0; i < cap; ++i)
                ramp[i] = static_cast<T>(i + ch * 1000);
            buf.writeAt(ch, 0, ramp.data(), cap);
            oracle.write(ch, 0, ramp.data(), cap);
        }
        buf.refreshMirror();

        std::vector<T> got, exp;
        for (int ch = 0; ch < 2; ++ch)
        {
            for (int start = 0; start < cap; ++start)
            {
                for (int len = 1; len <= kMaxWindow; ++len)
                {
                    got.resize(static_cast<std::size_t>(len));
                    exp.resize(static_cast<std::size_t>(len));
                    buf.readWindow(ch, start, got.data(), len);
                    oracle.read(ch, start, exp.data(), len);
                    for (int j = 0; j < len; ++j)
                        if (got[j] != exp[j])
                            FAIL("ch=%d start=%d len=%d j=%d got=%g exp=%g",
                                 ch, start, len, j, (double)got[j], (double)exp[j]);
                }
            }
        }
        std::printf("[%s] window read [1,%d]: PASS\n", name, kMaxWindow);
    }

    // ── 4. Contiguity ──────────────────────────────────────────────────────
    g_section = "contiguity";
    {
        Buf buf;
        buf.prepare(2, kTestCapacity);
        const int cap = buf.getCapacity();
        oracle.init(cap);

        std::vector<T> ramp(static_cast<std::size_t>(cap));
        for (int ch = 0; ch < 2; ++ch)
        {
            for (int i = 0; i < cap; ++i)
                ramp[i] = static_cast<T>(i + ch * 1000);
            buf.writeAt(ch, 0, ramp.data(), cap);
            oracle.write(ch, 0, ramp.data(), cap);
        }
        buf.refreshMirror();

        std::vector<T> exp;
        for (int ch = 0; ch < 2; ++ch)
        {
            for (int start = 0; start < cap; ++start)
            {
                for (int len = 1; len <= kMaxWindow; ++len)
                {
                    const T* p = buf.tryGetContiguous(ch, start, len);
                    if (p == nullptr)
                        FAIL("tryGetContiguous returned null ch=%d start=%d len=%d "
                             "(mirror=%d must cover kMaxWindow-1=%d)", ch, start, len, kMirror, kMaxWindow - 1);
                    exp.resize(static_cast<std::size_t>(len));
                    oracle.read(ch, start, exp.data(), len);
                    for (int j = 0; j < len; ++j)
                        if (p[j] != exp[j])
                            FAIL("contig mismatch ch=%d start=%d len=%d j=%d got=%g exp=%g",
                                 ch, start, len, j, (double)p[j], (double)exp[j]);
                }
            }
        }
        std::printf("[%s] contiguity (never null, matches model): PASS\n", name);
    }

    // ── 5. Interleaved sequence ────────────────────────────────────────────
    g_section = "interleaved";
    {
        Buf buf;
        buf.prepare(2, kTestCapacity);
        const int cap = buf.getCapacity();
        oracle.init(cap);

        Rng rng(20240725u);
        constexpr int kBlocks = 4000;
        std::vector<T> blk0, blk1, got, exp;
        for (int b = 0; b < kBlocks; ++b)
        {
            const int n = rng.range(1, kMaxBlock);
            blk0.resize(static_cast<std::size_t>(n));
            blk1.resize(static_cast<std::size_t>(n));
            const T base0 = static_cast<T>(b * 10 + 1);
            const T base1 = static_cast<T>(b * 10 + 2);
            for (int k = 0; k < n; ++k)
            {
                blk0[k] = base0 + static_cast<T>(k);
                blk1[k] = base1 + static_cast<T>(k);
            }
            const T* ptrs[2] = { blk0.data(), blk1.data() };

            const int expectedStart = oracle.writeIdx;
            const int startIdx = buf.pushBlock(ptrs, 2, n);
            if (startIdx != expectedStart)
                FAIL("block %d startIdx=%d expected=%d", b, startIdx, expectedStart);

            oracle.write(0, oracle.writeIdx, blk0.data(), n);   // channel 0
            oracle.write(1, oracle.writeIdx, blk1.data(), n);   // channel 1 (same start)
            oracle.advance(n);

            if (buf.getWriteIndex() != oracle.writeIdx)
                FAIL("block %d writeIdx=%d expected=%d", b, buf.getWriteIndex(), oracle.writeIdx);

            // Read a spread of windows: the just-written block, a random start,
            // and a start at the wrap neighbourhood behind the write head.
            const int readStarts[3] = {
                startIdx,
                rng.range(0, cap - 1),
                (startIdx + cap - kMaxWindow) % cap
            };
            for (int rs : readStarts)
            {
                for (int len = 1; len <= kMaxWindow; ++len)
                {
                    got.resize(static_cast<std::size_t>(len));
                    exp.resize(static_cast<std::size_t>(len));
                    for (int ch = 0; ch < 2; ++ch)
                    {
                        buf.readWindow(ch, rs, got.data(), len);
                        oracle.read(ch, rs, exp.data(), len);
                        for (int j = 0; j < len; ++j)
                            if (got[j] != exp[j])
                                FAIL("block %d read ch=%d rs=%d len=%d j=%d got=%g exp=%g",
                                     b, ch, rs, len, j, (double)got[j], (double)exp[j]);
                    }
                }
            }
        }
        std::printf("[%s] interleaved sequence (%d blocks): PASS\n", name, kBlocks);
    }

    // ── 6. Large capacity (1 << 18) ────────────────────────────────────────
    g_section = "large capacity";
    {
        Buf buf;
        buf.prepare(2, kLargeCap);
        const int cap = buf.getCapacity();
        CHECK(cap == kLargeCap);
        oracle.init(cap);

        // Fill with a unique ramp via a single full-capacity writeAt per channel.
        std::vector<T> ramp(static_cast<std::size_t>(cap));
        for (int ch = 0; ch < 2; ++ch)
        {
            for (int i = 0; i < cap; ++i)
                ramp[i] = static_cast<T>(i + ch * 1000);
            buf.writeAt(ch, 0, ramp.data(), cap);
            oracle.write(ch, 0, ramp.data(), cap);
        }
        buf.refreshMirror();

        // Start offsets clustered near the wrap (the last 64 indices) plus a
        // couple far from it, every legal window length. This is where an
        // overflow in startIndex + length or a mirror sized too small would
        // surface at realistic magnitude.
        std::vector<T> got, exp;
        const int clusterStart = cap - 64;
        for (int ch = 0; ch < 2; ++ch)
        {
            for (int start = clusterStart; start < cap; ++start)
            {
                for (int len = 1; len <= kMaxWindow; ++len)
                {
                    got.resize(static_cast<std::size_t>(len));
                    exp.resize(static_cast<std::size_t>(len));
                    buf.readWindow(ch, start, got.data(), len);
                    oracle.read(ch, start, exp.data(), len);
                    for (int j = 0; j < len; ++j)
                        if (got[j] != exp[j])
                            FAIL("ch=%d start=%d len=%d j=%d got=%g exp=%g",
                                 ch, start, len, j, (double)got[j], (double)exp[j]);

                    const T* p = buf.tryGetContiguous(ch, start, len);
                    if (p == nullptr)
                        FAIL("large-cap contig null ch=%d start=%d len=%d", ch, start, len);
                    for (int j = 0; j < len; ++j)
                        if (p[j] != exp[j])
                            FAIL("large-cap contig mismatch ch=%d start=%d len=%d j=%d", ch, start, len, j);
                }
            }
        }

        // Mirror invariant at scale: storage[cap + k] == storage[k].
        for (int ch = 0; ch < 2; ++ch)
        {
            const T* head = headPtr(buf, ch);
            const T* mir  = mirrorPtr(buf, ch);
            CHECK(head != nullptr && mir != nullptr);
            for (int k = 0; k < kMirror; ++k)
                if (mir[k] != head[k])
                    FAIL("large-cap mirror ch=%d k=%d mir=%g head=%g", ch, k, (double)mir[k], (double)head[k]);
        }
        std::printf("[%s] large capacity (1<<18), wrap-clustered: PASS\n", name);
    }

    return 0;
}

} // namespace

int main()
{
    std::printf("=== Chronos Pow2RingBuffer correctness harness ===\n");
    std::printf("mirror=%d  kMaxWindow=%d  testCapacity=%d  largeCapacity=%d\n\n",
                kMirror, kMaxWindow, kTestCapacity, kLargeCap);

    int r = 0;
    r |= runAll<float>("float");
    r |= runAll<double>("double");

    std::printf("\n=== %s ===\n", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}
