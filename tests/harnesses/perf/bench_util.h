// tests/harnesses/perf/bench_util.h
//
// Shared utilities for the perf harnesses:
//   setFtzDaz()  - flush-to-zero + denormals-are-zero, called once before timing
//                  so the harness matches the plugin's floating-point mode.
//   bench::Record / bench::writeJson - emit the --json baseline format that
//                  scripts/bench_gate.py compares.
//
// The ns_per_sample field is the only number bench_gate reads. The provisional
// flag marks a baseline the owner has not yet confirmed on a pinned core.
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace bench
{
    // Enable flush-to-zero and denormals-are-zero. Call once before the timed
    // region. On x86 this sets the MXCSR FTZ (bit 15) and DAZ (bit 6) flags. On
    // AArch64 the FPCR FZ bit (24) flushes both inputs and outputs. No-op on
    // architectures without a denormal control register.
    inline void setFtzDaz() noexcept
    {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        _mm_setcsr(_mm_getcsr() | 0x8040u);
#elif defined(__aarch64__) || defined(_M_ARM64)
        std::uint64_t fpcr = 0;
        __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
        fpcr |= (1ull << 24);
        __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
#endif
    }

    struct Record
    {
        std::string name;
        std::string config;
        double ns_per_sample;
    };

    // Escape a string for a JSON string literal. The bench names and config
    // strings are simple (alphanumeric, =, +, <>, ,) so this is a safety net.
    inline std::string jsonEscape(const std::string &s)
    {
        std::string out;
        out.reserve(s.size());
        for (char c: s)
        {
            if (c == '"') out += "\\\"";
            else if (c == '\\') out += "\\\\";
            else out += c;
        }
        return out;
    }

    // Write records as the bench_gate baseline JSON. Creates parent directories.
    inline void writeJson(const std::string &path, const std::vector<Record> &records, bool provisional)
    {
        const std::filesystem::path p(path);
        if (p.has_parent_path())
            std::filesystem::create_directories(p.parent_path());
        std::ofstream f(p, std::ios::trunc);
        f << "{\n  \"provisional\": " << (provisional ? "true" : "false") << ",\n  \"records\": [\n";
        for (std::size_t i = 0; i < records.size(); ++i)
        {
            const Record &r = records[i];
            f << "    {\"name\": \"" << jsonEscape(r.name) << "\", \"config\": \""
                    << jsonEscape(r.config) << "\", \"ns_per_sample\": " << r.ns_per_sample << "}";
            if (i + 1 < records.size()) f << ",";
            f << "\n";
        }
        f << "  ]\n}\n";
        f.flush();
    }
} // namespace bench
