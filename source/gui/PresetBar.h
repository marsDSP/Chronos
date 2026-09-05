#pragma once

#ifndef CHRONOS_PRESET_BAR_H
#define CHRONOS_PRESET_BAR_H

#include <JuceHeader.h>
#include <functional>
#include "Colours.h"
#include "Metrics.h"
#include "AccentConsumer.h"
#include "../presets/PresetStore.h"

namespace MarsDSP::Presets { class PresetManager; }

namespace MarsDSP::GUI {

// The header preset browser. Shows the current preset name,
// step arrows, and a menu of preset file actions.
class PresetBar : public Component,
                  public AccentConsumer,
                  public SettableTooltipClient,
                  private Timer {
public:
    explicit PresetBar(MarsDSP::Presets::PresetManager& pm);
    ~PresetBar() override;

    // Set the scale metrics for the bar layout.
    void setMetrics(const Metrics& m);

    // Store the live core accent for the border and the modified ink.
    void setAccentColour(Colour c) override;

    void paint(Graphics& g) override;
    void resized() override;
    void mouseMove(const MouseEvent& e) override;
    void mouseExit(const MouseEvent& e) override;
    void mouseUp(const MouseEvent& e) override;

    // Return a region-specific tooltip for the pointer position.
    String getTooltip() override;

private:
    void timerCallback() override;
    void refreshName_();
    void showMenu_();
    void handleMenuResult_(int result);
    void stepPreset_(int direction);

    // Confirm before a load discards unsaved edits. Every load path
    // routes through this guard, and proceed runs only on yes.
    void confirmDiscardChanges_(std::function<void()> proceed);

    void doSave_();
    void doSaveAs_();
    void completeSaveAs_(const String& name);
    void doRename_();
    void completeRename_(const String& name);
    void doDelete_();
    void completeDelete_();
    void doExport_();
    void doImport_();

    MarsDSP::Presets::PresetManager& pm_;
    Metrics metrics_;
    Colour accent_ { Colours::accentDelayDigital };
    String displayedName_;
    bool modifiedSeen_ = false;
    int hoveredRegion_ = 0; // 0 none, 1 prev, 2 name, 3 next, 4 menu
    Point<int> lastMousePos_;

    std::vector<File> menuPresetFiles_;
    std::vector<MarsDSP::Presets::PresetEntry> menuFactoryPresets_;
    std::unique_ptr<FileChooser> exportChooser_;
    std::unique_ptr<FileChooser> importChooser_;

    // Hit regions in pixels, set in resized().
    Rectangle<int> prevArea_, nameArea_, nextArea_, menuArea_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PresetBar)
};

} // namespace MarsDSP::GUI

#endif
