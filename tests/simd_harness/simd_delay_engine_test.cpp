#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <sstream>
#include "dsp/engine/delay/delay_engine.h"

using namespace MarsDSP::DSP;

struct TestResult {
    std::string name;
    std::vector<float> input;
    std::vector<float> scalarOut;
    std::vector<float> simdOut;
    std::vector<float> error;
    double maxAbsError;
    double mae;
    double rmse;
};

// Helper to generate SVG with multiple informative graphics
void generateDetailedSVG(const std::string& filename, 
                         const std::vector<TestResult>& results,
                         int width = 1200, int height = 1600)
{
    std::ofstream svg(filename);
    if (!svg.is_open()) return;

    svg << "<svg width=\"" << width << "\" height=\"" << height << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
    svg << "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n";
    
    // Title
    svg << "<text x=\"" << width/2 << "\" y=\"40\" font-family=\"sans-serif\" font-size=\"24\" font-weight=\"bold\" text-anchor=\"middle\" fill=\"#333\">DelayEngine SIMD Correctness Report</text>\n";

    float currentY = 80;
    float margin = 60;
    float chartWidth = width - 2 * margin;
    float chartHeight = 180;

    for (const auto& res : results) {
        svg << "<text x=\"" << margin << "\" y=\"" << currentY << "\" font-family=\"sans-serif\" font-size=\"20\" font-weight=\"bold\" fill=\"#2980b9\">Mode: " << res.name << "</text>\n";
        currentY += 30;

        auto plotArea = [&](float x, float y, float w, float h, const std::string& title) {
            svg << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << w << "\" height=\"" << h << "\" fill=\"#fdfdfd\" stroke=\"#eee\"/>\n";
            svg << "<text x=\"" << x + 10 << "\" y=\"" << y + 20 << "\" font-family=\"sans-serif\" font-size=\"14\" font-weight=\"bold\" fill=\"#777\">" << title << "</text>\n";
            svg << "<line x1=\"" << x << "\" y1=\"" << y + h/2 << "\" x2=\"" << x + w << "\" y2=\"" << y + h/2 << "\" stroke=\"#ddd\" stroke-dasharray=\"4\"/>\n";
        };

        // 1. Reference Output Plot
        plotArea(margin, currentY, chartWidth, chartHeight, "Reference Output (Scalar)");
        svg << "<path d=\"M " << margin << " " << currentY + chartHeight/2 << " ";
        for (size_t i = 0; i < res.scalarOut.size(); ++i) {
            float px = margin + (float)i / res.scalarOut.size() * chartWidth;
            float py = (currentY + chartHeight/2) - res.scalarOut[i] * (chartHeight * 0.45f);
            svg << "L " << px << " " << py << " ";
        }
        svg << "\" fill=\"none\" stroke=\"#2ecc71\" stroke-width=\"1\"/>\n";
        currentY += chartHeight + 20;

        // 2. SIMD Error Plot
        plotArea(margin, currentY, chartWidth, chartHeight, "SIMD vs Scalar Error (Magnified)");
        float errScale = (res.maxAbsError > 1e-15f) ? (chartHeight * 0.4f / (float)res.maxAbsError) : 1.0f;
        svg << "<path d=\"M " << margin << " " << currentY + chartHeight/2 << " ";
        for (size_t i = 0; i < res.error.size(); ++i) {
            float px = margin + (float)i / res.error.size() * chartWidth;
            float py = (currentY + chartHeight/2) - res.error[i] * errScale;
            svg << "L " << px << " " << py << " ";
        }
        svg << "\" fill=\"none\" stroke=\"#e74c3c\" stroke-width=\"1\"/>\n";
        svg << "<text x=\"" << margin + chartWidth - 10 << "\" y=\"" << currentY + 20 << "\" font-family=\"monospace\" font-size=\"12\" text-anchor=\"end\" fill=\"#e74c3c\">Scale: " << std::scientific << (double)errScale << "x</text>\n";
        currentY += chartHeight + 30;

        // 3. Stats & Histogram
        float statsH = 150;
        plotArea(margin, currentY, chartWidth * 0.4f, statsH, "Statistics");
        
        auto drawStat = [&](int row, const std::string& label, double value, bool isScientific = true) {
            float ry = currentY + 45 + row * 22;
            svg << "<text x=\"" << margin + 20 << "\" y=\"" << ry << "\" font-family=\"sans-serif\" font-size=\"13\" fill=\"#666\">" << label << ":</text>\n";
            std::stringstream ss;
            if (isScientific) ss << std::scientific << std::setprecision(4);
            else ss << std::fixed << std::setprecision(0);
            ss << value;
            svg << "<text x=\"" << margin + 180 << "\" y=\"" << ry << "\" font-family=\"monospace\" font-size=\"13\" font-weight=\"bold\" fill=\"#333\">" << ss.str() << "</text>\n";
        };

        drawStat(0, "Samples Processed", (double)res.error.size(), false);
        drawStat(1, "Max Abs Error", res.maxAbsError);
        drawStat(2, "Mean Abs Error", res.mae);
        drawStat(3, "Root Mean Sq Err", res.rmse);
        
        float ry_status = currentY + 45 + 4 * 22;
        svg << "<text x=\"" << margin + 20 << "\" y=\"" << ry_status << "\" font-family=\"sans-serif\" font-size=\"13\" fill=\"#666\">Status:</text>\n";
        svg << "<text x=\"" << margin + 180 << "\" y=\"" << ry_status << "\" font-family=\"sans-serif\" font-size=\"13\" font-weight=\"bold\" fill=\"" << (res.maxAbsError < 1e-5 ? "#2ecc71" : "#e74c3c") << "\">" << (res.maxAbsError < 1e-5 ? "PASS" : "FAIL") << "</text>\n";

        // Histogram
        plotArea(margin + chartWidth * 0.45f, currentY, chartWidth * 0.55f, statsH, "Error Magnitude Distribution");
        const int numBins = 40;
        std::vector<int> bins(numBins, 0);
        for (float e : res.error) {
            float norm = res.maxAbsError > 0 ? std::abs(e) / (float)res.maxAbsError : 0;
            int bin = std::clamp((int)(norm * (numBins - 1)), 0, numBins - 1);
            bins[bin]++;
        }
        int maxBin = *std::max_element(bins.begin(), bins.end());
        float binWidth = (chartWidth * 0.55f - 40) / numBins;
        for (int i = 0; i < numBins; ++i) {
            float h = maxBin > 0 ? (float)bins[i] / maxBin * (statsH - 50) : 0;
            svg << "<rect x=\"" << margin + chartWidth * 0.45f + 20 + i * binWidth << "\" y=\"" << currentY + statsH - 10 - h << "\" width=\"" << binWidth - 1 << "\" height=\"" << h << "\" fill=\"#3498db\"/>\n";
        }
        
        currentY += statsH + 60;
    }

    svg << "</svg>\n";
    svg.close();
}

TestResult runTest(const std::string& name, bool isMono)
{
    const int sampleRate = 44100;
    const int numSamples = 2048;
    const float delayMs = isMono ? 12.34f : 45.67f;
    const float mix = 0.7f;
    const float feedback = 0.6f;

    DelayEngine<float> engineSIMD;
    DelayEngine<float> engineScalar;

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = 4096;
    spec.numChannels = isMono ? 1 : 2;

    engineSIMD.prepare(spec);
    engineScalar.prepare(spec);

    engineSIMD.setDelayTimeParam(delayMs);
    engineSIMD.setMixParam(mix);
    engineSIMD.setFeedbackParam(feedback);
    engineSIMD.setMono(isMono);

    engineScalar.setDelayTimeParam(delayMs);
    engineScalar.setMixParam(mix);
    engineScalar.setFeedbackParam(feedback);
    engineScalar.setMono(isMono);

    std::vector<float> input(numSamples);
    std::mt19937 gen(isMono ? 42 : 123);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < numSamples; ++i) input[i] = (i % 256 == 0) ? 1.0f : dis(gen) * 0.05f;

    // Output L channel for comparison
    std::vector<float> outputSIMD(numSamples);
    std::vector<float> outputScalar(numSamples);

    // Process SIMD
    juce::AudioBuffer<float> bufferSIMD(spec.numChannels, numSamples);
    for (int ch = 0; ch < spec.numChannels; ++ch) 
        for (int i = 0; i < numSamples; ++i) bufferSIMD.setSample(ch, i, input[i]);
    
    juce::dsp::AudioBlock<float> blockSIMD(bufferSIMD);
    engineSIMD.process(blockSIMD, numSamples);
    for (int i = 0; i < numSamples; ++i) outputSIMD[i] = bufferSIMD.getSample(0, i);

    // Process Scalar
    for (int i = 0; i < numSamples; ++i) {
        juce::AudioBuffer<float> bufferScalar(spec.numChannels, 1);
        for (int ch = 0; ch < spec.numChannels; ++ch) bufferScalar.setSample(ch, 0, input[i]);
        juce::dsp::AudioBlock<float> blockScalar(bufferScalar);
        engineScalar.process(blockScalar, 1);
        outputScalar[i] = bufferScalar.getSample(0, 0);
    }

    TestResult res;
    res.name = name;
    res.input = input;
    res.scalarOut = outputScalar;
    res.simdOut = outputSIMD;
    res.error.resize(numSamples);
    res.maxAbsError = 0;
    double sumErr = 0;
    double sumSqErr = 0;

    for (int i = 0; i < numSamples; ++i) {
        res.error[i] = outputSIMD[i] - outputScalar[i];
        double absE = std::abs(res.error[i]);
        res.maxAbsError = std::max(res.maxAbsError, absE);
        sumErr += absE;
        sumSqErr += absE * absE;
    }
    res.mae = sumErr / numSamples;
    res.rmse = std::sqrt(sumSqErr / numSamples);
    
    return res;
}

int main()
{
    std::cout << "Starting DelayEngine SIMD Correctness Suite..." << std::endl;
    
    std::vector<TestResult> results;
    results.push_back(runTest("Mono Path", true));
    results.push_back(runTest("Stereo Path (L-Channel)", false));

    for (const auto& res : results) {
        std::cout << "[" << res.name << "] Max Error: " << std::scientific << res.maxAbsError << std::endl;
    }

    generateDetailedSVG("tests/simd_harness/logs/simd_delay_report.svg", results);
    std::cout << "Detailed SVG report generated: tests/simd_harness/logs/simd_delay_report.svg" << std::endl;

    for (const auto& res : results) {
        if (res.maxAbsError > 1e-5) return 1;
    }
    return 0;
}
