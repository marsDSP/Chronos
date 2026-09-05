#pragma once

#ifndef CHRONOS_TIME_DISPLAY_FORMATTER_H
#define CHRONOS_TIME_DISPLAY_FORMATTER_H

#include <JuceHeader.h>

namespace MarsDSP::GUI::TimeDisplayFormatter {

// Format delay time in milliseconds or tempo-sync division text.
// divisionIndex picks the division name when syncActive is true.
String getDelayTimeText(const Slider* slider, bool syncActive, int divisionIndex);

} // namespace MarsDSP::GUI::TimeDisplayFormatter

#endif
