/**
 * Correctness harness for SpscFifo.
 * Validates single-thread fill and drain exactness.
 * Validates wraparound endurance and queue bounds.
 * Validates concurrent producer and consumer soak.
 */

#include "utils/memory/SpscFifo.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <print>
#include <thread>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

struct TestPayload {
    std::uint64_t seq = 0;
    std::uint64_t data = 0;
};

int runAll()
{
    using MarsDSP::Memory::SpscFifo;

    // 1. Single-thread fill and drain exactness.
    g_section = "fill and drain";
    {
        constexpr std::size_t kCap = 16;
        SpscFifo<int, kCap> fifo;

        int out = -1;
        CHECK(!fifo.pop(out));

        for (int i = 0; i < static_cast<int>(kCap); ++i)
        {
            const bool ok = fifo.push(i);
            if (!ok) FAIL("push failed at index {}", i);
        }

        CHECK(!fifo.push(999));

        for (int i = 0; i < static_cast<int>(kCap); ++i)
        {
            const bool ok = fifo.pop(out);
            if (!ok) FAIL("pop failed at index {}", i);
            if (out != i) FAIL("expected {} got {}", i, out);
        }

        CHECK(!fifo.pop(out));
        std::println("fill and drain: PASS");
    }

    // 2. Wraparound endurance.
    g_section = "wraparound endurance";
    {
        constexpr std::size_t kCap = 8;
        SpscFifo<std::uint32_t, kCap> fifo;

        std::uint32_t pushVal = 0;
        std::uint32_t popVal = 0;
        constexpr std::uint32_t kCycles = 10000;

        for (std::uint32_t c = 0; c < kCycles; ++c)
        {
            for (int i = 0; i < 5; ++i)
            {
                CHECK(fifo.push(pushVal++));
            }
            for (int i = 0; i < 5; ++i)
            {
                std::uint32_t val = 0;
                CHECK(fifo.pop(val));
                if (val != popVal)
                    FAIL("wraparound cycle {} expected {} got {}", c, popVal, val);
                ++popVal;
            }
        }

        std::uint32_t dummy = 0;
        CHECK(!fifo.pop(dummy));
        std::println("wraparound endurance ({} cycles): PASS", kCycles);
    }

    // 3. Queue bounds and clear.
    g_section = "bounds and clear";
    {
        constexpr std::size_t kCap = 4;
        SpscFifo<int, kCap> fifo;

        int out = 0;
        CHECK(!fifo.pop(out));

        CHECK(fifo.push(1));
        CHECK(fifo.push(2));
        CHECK(fifo.push(3));
        CHECK(fifo.push(4));
        CHECK(!fifo.push(5));

        CHECK(fifo.pop(out) && out == 1);
        CHECK(fifo.push(5));
        CHECK(!fifo.push(6));

        fifo.clear();
        CHECK(!fifo.pop(out));

        CHECK(fifo.push(10));
        CHECK(fifo.push(20));
        CHECK(fifo.push(30));
        CHECK(fifo.push(40));
        CHECK(!fifo.push(50));

        CHECK(fifo.pop(out) && out == 10);
        CHECK(fifo.pop(out) && out == 20);
        CHECK(fifo.pop(out) && out == 30);
        CHECK(fifo.pop(out) && out == 40);
        CHECK(!fifo.pop(out));

        std::println("bounds and clear: PASS");
    }

    // 4. Two-thread producer and consumer soak.
    g_section = "concurrent soak";
    {
        constexpr std::size_t kCap = 256;
        constexpr std::uint64_t kTotalItems = 1000000;

        SpscFifo<TestPayload, kCap> fifo;
        std::atomic<bool> producerDone{false};
        std::atomic<std::uint64_t> itemsReceived{0};
        std::atomic<bool> sequenceValid{true};

        std::thread consumer([&fifo, &producerDone, &itemsReceived, &sequenceValid]() {
            std::uint64_t expectedSeq = 0;
            TestPayload item{};

            while (true)
            {
                if (fifo.pop(item))
                {
                    if (item.seq != expectedSeq || item.data != ~expectedSeq)
                    {
                        sequenceValid.store(false, std::memory_order_relaxed);
                    }
                    ++expectedSeq;
                    itemsReceived.fetch_add(1, std::memory_order_relaxed);
                }
                else
                {
                    if (producerDone.load(std::memory_order_acquire))
                    {
                        while (fifo.pop(item))
                        {
                            if (item.seq != expectedSeq || item.data != ~expectedSeq)
                            {
                                sequenceValid.store(false, std::memory_order_relaxed);
                            }
                            ++expectedSeq;
                            itemsReceived.fetch_add(1, std::memory_order_relaxed);
                        }
                        break;
                    }
                    std::this_thread::yield();
                }
            }
        });

        std::thread producer([&fifo, &producerDone]() {
            for (std::uint64_t i = 0; i < kTotalItems; ++i)
            {
                TestPayload item{i, ~i};
                while (!fifo.push(item))
                {
                    std::this_thread::yield();
                }
            }
            producerDone.store(true, std::memory_order_release);
        });

        producer.join();
        consumer.join();

        CHECK(sequenceValid.load());
        CHECK(itemsReceived.load() == kTotalItems);

        std::println("concurrent soak ({} items): PASS", kTotalItems);
    }

    return 0;
}

} // namespace

int main()
{
    std::println("=== Chronos SpscFifo correctness harness ===");
    std::println();

    const int r = runAll();

    std::println();
    std::println("=== {} ===", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}
