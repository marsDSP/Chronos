#pragma once

#ifndef CHRONOS_PRESET_MANAGER_H
#define CHRONOS_PRESET_MANAGER_H

#include <JuceHeader.h>
#include <atomic>
#include "PresetStore.h"

class ChronosProcessor;

namespace MarsDSP::Presets {

// The policy layer. Owns a PresetStore, holds the current
// preset identity, tracks the modified flag, and performs
// load and save against the processor.
class PresetManager final : private AudioProcessorValueTreeState::Listener {
public:
    PresetManager(AudioProcessor& proc, AudioProcessorValueTreeState& apvts);
    explicit PresetManager(ChronosProcessor& proc);
    ~PresetManager() override;

    // The current preset name. Empty when no preset is loaded.
    String getCurrentName() const { return presetName_; }
    String getCurrentBank() const { return presetBank_; }
    bool isCurrentFactory() const { return isFactory_; }

    // Return true when a parameter moved since the last load or save.
    bool isModified() const { return modified_.load(std::memory_order_relaxed); }

    // Clear the modified flag.
    void clearModified() { modified_.store(false, std::memory_order_relaxed); }

    // Save the current processor state as a new preset file.
    // Return true on success.
    bool saveAs(const String& name, const String& author, const String& category);

    // Save over the current preset file. Refuse when the preset is factory.
    bool saveCurrent(const String& author, const String& category);

    // Load a preset file through the processor state recall path.
    // Return true on success. On failure leave the state untouched.
    bool loadPreset(const File& file);

    // Load the preset identity from a file without changing the state.
    void loadIdentity(const File& file);

    // Delete the current preset file. Refuse when the preset is factory.
    bool deleteCurrent();

    // Rename the current preset file. Refuse when factory.
    bool renameCurrent(const String& newName);

    // The preset store.
    PresetStore& getStore() { return store_; }

    // The list of user presets.
    std::vector<PresetEntry> getUserPresets() const { return store_.enumerateUserPresets(); }

    // Copy the current preset XML to a string.
    String copyPresetXml();

    // Paste a preset XML string and load it as an unnamed modified patch.
    // Return true on success.
    bool pastePresetXml(const String& xmlText);

private:
    void parameterChanged(const String& parameterID, float newValue) override;

    void registerParameterListeners_();
    void unregisterParameterListeners_();

    // Apply a state tree through the processor recall path.
    bool applyStateXml_(const XmlElement& xml);

    // Reject a preset with an unknown parameter or an out-of-range value.
    bool validatePresetXml_(const XmlElement& xml);

    AudioProcessor& processorRef_;
    AudioProcessorValueTreeState& apvtsRef_;
    PresetStore store_;
    String presetName_;
    String presetBank_;
    String presetAuthor_;
    String presetCategory_;
    bool isFactory_ = false;
    File currentFile_;
    std::atomic<bool> modified_ { false };
};

} // namespace MarsDSP::Presets

#endif
