#pragma once

#ifndef CHRONOS_TAP_FEED_FRAME_H
#define CHRONOS_TAP_FEED_FRAME_H

namespace MarsDSP::GUI {

// Audio level frame sent from the audio thread to the user interface.
struct TapFeedFrame {
    float rmsL = 0.0f;
    float rmsR = 0.0f;
    float wetRmsL = 0.0f;
    float wetRmsR = 0.0f;
};

} // namespace MarsDSP::GUI

#endif
