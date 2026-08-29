#pragma once

#ifndef CHRONOS_TIME_DISPLAY_FORMATTER_H
#define CHRONOS_TIME_DISPLAY_FORMATTER_H

#include <JuceHeader.h>

namespace MarsDSP::GUI::TimeDisplayFormatter {

// Format delay time in milliseconds or tempo-sync division text.
String getDelayTimeText(const Slider* slider, bool syncActive);

} // namespace MarsDSP::GUI::TimeDisplayFormatter

#endif
