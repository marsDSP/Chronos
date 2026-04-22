// Performance harness for AlignedSIMDBuffer vs std::vector<float> and a
// bare float[] baseline. Measures:
//   * sequential write-then-read throughput
//   * whole-channel memcpy throughput
// across a sweep of channel sample counts.
//
// Emits tests/perf_harness/logs/perf_aligned_buffer.csv.

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "dsp/buffers/aligned_simd_buffer.h"

namespace
{
    template <typename Fn>
    double timeAverageMicrosecondsPerIteration(int numIterations, Fn&& fn)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        for (int iterationIndex = 0; iterationIndex < numIterations; ++iterationIndex)
            fn();
        const auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start).count()
               / numIterations;
    }
}

int main()
{
    std::filesystem::create_directories("tests/perf_harness/logs");

    const std::vector<int> perChannelSampleCountSweep = {
        256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
    };
    constexpr int numIterations = 20000;

    std::ofstream csvSink("tests/perf_harness/logs/perf_aligned_buffer.csv");
    csvSink << "container,num_samples,write_read_us,memcpy_us\n";
    csvSink << std::fixed << std::setprecision(6);

    std::cout << "[perf aligned_buffer] write+read and memcpy throughput\n";

    for (int numSamplesPerChannel : perChannelSampleCountSweep)
    {
        // ---- float[] (raw heap) ------------------------------------
        std::unique_ptr<float[]> rawFloatHeapBuffer(
            new float[static_cast<size_t>(numSamplesPerChannel)]);
        std::unique_ptr<float[]> rawFloatHeapBufferCopyTarget(
            new float[static_cast<size_t>(numSamplesPerChannel)]);

        const double rawWriteReadUs = timeAverageMicrosecondsPerIteration(
            numIterations,
            [&]
            {
                for (int i = 0; i < numSamplesPerChannel; ++i)
                    rawFloatHeapBuffer[i] =
                        static_cast<float>(i & 0xFFFF) * 0.001f;
                float accumulatedReadSum = 0.0f;
                for (int i = 0; i < numSamplesPerChannel; ++i)
                    accumulatedReadSum += rawFloatHeapBuffer[i];
                if (accumulatedReadSum > 1e30f) std::cout << "never\n";
            });

        const double rawMemcpyUs = timeAverageMicrosecondsPerIteration(
            numIterations,
            [&]
            {
                std::memcpy(rawFloatHeapBufferCopyTarget.get(),
                            rawFloatHeapBuffer.get(),
                            sizeof(float)
                                * static_cast<size_t>(numSamplesPerChannel));
            });

        // ---- std::vector<float> ------------------------------------
        std::vector<float> stdVectorBuffer(
            static_cast<size_t>(numSamplesPerChannel));
        std::vector<float> stdVectorBufferCopyTarget(
            static_cast<size_t>(numSamplesPerChannel));

        const double stdVectorWriteReadUs = timeAverageMicrosecondsPerIteration(
            numIterations,
            [&]
            {
                for (int i = 0; i < numSamplesPerChannel; ++i)
                    stdVectorBuffer[static_cast<size_t>(i)] =
                        static_cast<float>(i & 0xFFFF) * 0.001f;
                float accumulatedReadSum = 0.0f;
                for (int i = 0; i < numSamplesPerChannel; ++i)
                    accumulatedReadSum +=
                        stdVectorBuffer[static_cast<size_t>(i)];
                if (accumulatedReadSum > 1e30f) std::cout << "never\n";
            });

        const double stdVectorMemcpyUs = timeAverageMicrosecondsPerIteration(
            numIterations,
            [&]
            {
                std::memcpy(stdVectorBufferCopyTarget.data(),
                            stdVectorBuffer.data(),
                            sizeof(float)
                                * static_cast<size_t>(numSamplesPerChannel));
            });

        // ---- AlignedSIMDBuffer (single channel) --------------------
        MarsDSP::DSP::AlignedBuffers::AlignedSIMDBuffer<float>
            alignedSIMDBuffer(1, numSamplesPerChannel);
        MarsDSP::DSP::AlignedBuffers::AlignedSIMDBuffer<float>
            alignedSIMDBufferCopyTarget(1, numSamplesPerChannel);

        const double alignedSIMDWriteReadUs = timeAverageMicrosecondsPerIteration(
            numIterations,
            [&]
            {
                float* const channelBasePointer =
                    alignedSIMDBuffer.getWritePointer(0);
                for (int i = 0; i < numSamplesPerChannel; ++i)
                    channelBasePointer[i] =
                        static_cast<float>(i & 0xFFFF) * 0.001f;
                float accumulatedReadSum = 0.0f;
                const float* const readPointer =
                    alignedSIMDBuffer.getReadPointer(0);
                for (int i = 0; i < numSamplesPerChannel; ++i)
                    accumulatedReadSum += readPointer[i];
                if (accumulatedReadSum > 1e30f) std::cout << "never\n";
            });

        const double alignedSIMDMemcpyUs = timeAverageMicrosecondsPerIteration(
            numIterations,
            [&]
            {
                std::memcpy(alignedSIMDBufferCopyTarget.getWritePointer(0),
                            alignedSIMDBuffer.getReadPointer(0),
                            sizeof(float)
                                * static_cast<size_t>(numSamplesPerChannel));
            });

        csvSink << "float[]," << numSamplesPerChannel << ","
                << rawWriteReadUs << "," << rawMemcpyUs << "\n";
        csvSink << "std::vector<float>," << numSamplesPerChannel << ","
                << stdVectorWriteReadUs << "," << stdVectorMemcpyUs << "\n";
        csvSink << "AlignedSIMDBuffer," << numSamplesPerChannel << ","
                << alignedSIMDWriteReadUs << "," << alignedSIMDMemcpyUs << "\n";

        std::cout << "  N=" << std::setw(5) << numSamplesPerChannel
                  << "  raw write+read="      << rawWriteReadUs
                  << "us  vec=" << stdVectorWriteReadUs
                  << "us  aligned=" << alignedSIMDWriteReadUs << "us\n";
    }

    csvSink.close();
    return 0;
}
