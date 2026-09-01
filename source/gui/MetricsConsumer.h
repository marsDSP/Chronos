#pragma once

#ifndef CHRONOS_METRICS_CONSUMER_H
#define CHRONOS_METRICS_CONSUMER_H

#include <JuceHeader.h>
#include "Metrics.h"

namespace MarsDSP::GUI {

// Interface for components that receive the editor scale metrics.
// The editor calls setMetrics on every consumer.
struct MetricsConsumer {
    virtual ~MetricsConsumer() = default;

    // Store the metrics and relayout.
    virtual void setMetrics(const Metrics&) = 0;
};

} // namespace MarsDSP::GUI

#endif
