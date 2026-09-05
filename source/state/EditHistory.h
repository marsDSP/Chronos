#pragma once

#ifndef CHRONOS_EDIT_HISTORY_H
#define CHRONOS_EDIT_HISTORY_H

#include <JuceHeader.h>
#include <array>
#include <functional>
#include <vector>

namespace MarsDSP::State {

// A gesture-scoped undo and redo history. Not the JUCE UndoManager:
// the APVTS undo manager records host automation and clears on preset
// load. This history records exactly the user edits, groups an XY drag
// into one step, and makes a preset load undoable.
class EditHistory : private AudioProcessorParameter::Listener {
public:
    // The maximum parameter count an entry can hold.
    static constexpr int kMaxParams = 32;
    // The ring capacity.
    static constexpr int kCapacity = 100;

    explicit EditHistory(AudioProcessor& proc);
    ~EditHistory() override;

    [[nodiscard]] bool canUndo() const noexcept;
    [[nodiscard]] bool canRedo() const noexcept;
    [[nodiscard]] String undoName() const;
    [[nodiscard]] String redoName() const;

    void undo();
    void redo();

    // Read the current value of every parameter as after, pair it with
    // before, and commit. Drop entries where before == after.
    void recordSnapshot(const String& name, const std::vector<float>& beforeNormalised);

    void clear();

    // The editor sets this to update the undo and redo buttons.
    std::function<void()> onChanged;

private:
    struct Change {
        int paramIndex = 0;
        float before = 0.0f;
        float after = 0.0f;
    };

    struct Entry {
        std::array<Change, kMaxParams> changes {};
        int count = 0;
        // The first parameter index, for the name.
        int firstParam = -1;
    };

    void parameterValueChanged(int, float) override {}
    void parameterGestureChanged(int index, bool starting) override;

    void commit_(const Entry& e, const String& name);
    void notifyChanged_();
    [[nodiscard]] String entryName_(const Entry& e) const;

    AudioProcessor& processorRef_;
    std::vector<AudioProcessorParameter*> params_;

    // The ring.
    std::array<Entry, kCapacity> entries_ {};
    int cursor_ = 0;   // where the next commit writes
    int count_ = 0;    // total valid entries
    int undoCount_ = 0; // entries before the cursor (undoable)

    // Grouping state.
    Entry pending_;
    int openCount_ = 0;
    bool applying_ = false;
};

} // namespace MarsDSP::State

#endif
