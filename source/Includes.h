#pragma once

#ifndef SOJOURNER_INCLUDES_H
#define SOJOURNER_INCLUDES_H

#ifndef DONT_SET_USING_JUCE_NAMESPACE
#define DONT_SET_USING_JUCE_NAMESPACE 1
#endif

#include <cmath>
#include <memory>
#include <array>
#include <vector>
#include <juce_audio_plugin_client/juce_audio_plugin_client.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_extra/juce_gui_extra.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_graphics/juce_graphics.h>
#include <juce_events/juce_events.h>
#include <juce_core/juce_core.h>
#include <juce_data_structures/juce_data_structures.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_product_unlocking/juce_product_unlocking.h>
#include <juce_cryptography/juce_cryptography.h>

#if JUCE_TARGET_HAS_BINARY_DATA
 #include "BinaryData.h"
#endif

#endif
