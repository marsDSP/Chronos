#include "TimeDisplayFormatter.h"

namespace MarsDSP::GUI::TimeDisplayFormatter {

String getDelayTimeText(const Slider* slider, const bool syncActive, const int divisionIndex)
{
    if (slider == nullptr)
        return "---";

    const double val = slider->getValue();

    if (syncActive)
    {
        static const StringArray divisions = {
            "1/64", "1/32T", "1/32", "1/16T", "1/32.", "1/16",
            "1/8T", "1/16.", "1/8", "1/4T", "1/8.", "1/4",
            "1/2T", "1/4.", "1/2", "1/1T", "1/2.", "1/1",
            "2/1", "4/1"
        };

        if (divisionIndex >= 0 && divisionIndex < divisions.size())
            return divisions[divisionIndex];

        return "1/4";
    }

    if (val < 1000.0)
        return String(roundToInt(val)) + " ms";

    return String(roundToInt(val * 0.001 * 100.0) / 100.0) + " s";
}

} // namespace MarsDSP::GUI::TimeDisplayFormatter
