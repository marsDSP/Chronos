#include "EditHistory.h"

#include <algorithm>

namespace MarsDSP::State {

EditHistory::EditHistory(AudioProcessor& proc)
    : processorRef_(proc)
{
    // Cache the parameter pointers and add this as a listener on each.
    const auto& ps = proc.getParameters();
    for (auto* p : ps)
    {
        params_.push_back(p);
        p->addListener(this);
    }
}

EditHistory::~EditHistory()
{
    for (auto* p : params_)
        p->removeListener(this);
}

bool EditHistory::canUndo() const noexcept
{
    return undoCount_ > 0;
}

bool EditHistory::canRedo() const noexcept
{
    return undoCount_ < count_;
}

String EditHistory::undoName() const
{
    if (! canUndo())
        return {};
    const int idx = (cursor_ - 1 + kCapacity) % kCapacity;
    return entryName_(entries_[static_cast<std::size_t>(idx)]);
}

String EditHistory::redoName() const
{
    if (! canRedo())
        return {};
    return entryName_(entries_[static_cast<std::size_t>(cursor_)]);
}

void EditHistory::undo()
{
    if (! canUndo())
        return;

    cursor_ = (cursor_ - 1 + kCapacity) % kCapacity;
    --undoCount_;

    applying_ = true;
    const auto& e = entries_[static_cast<std::size_t>(cursor_)];
    // Apply the before values in reverse order.
    for (int i = e.count - 1; i >= 0; --i)
    {
        const auto& c = e.changes[static_cast<std::size_t>(i)];
        auto* p = params_[static_cast<std::size_t>(c.paramIndex)];
        p->beginChangeGesture();
        p->setValueNotifyingHost(c.before);
        p->endChangeGesture();
    }
    applying_ = false;

    notifyChanged_();
}

void EditHistory::redo()
{
    if (! canRedo())
        return;

    const auto& e = entries_[static_cast<std::size_t>(cursor_)];

    applying_ = true;
    for (int i = 0; i < e.count; ++i)
    {
        const auto& c = e.changes[static_cast<std::size_t>(i)];
        auto* p = params_[static_cast<std::size_t>(c.paramIndex)];
        p->beginChangeGesture();
        p->setValueNotifyingHost(c.after);
        p->endChangeGesture();
    }
    applying_ = false;

    cursor_ = (cursor_ + 1) % kCapacity;
    ++undoCount_;

    notifyChanged_();
}

void EditHistory::recordSnapshot(const String& name, const std::vector<float>& beforeNormalised)
{
    Entry e;
    const int n = std::min(static_cast<int>(beforeNormalised.size()), static_cast<int>(params_.size()));
    for (int i = 0; i < n; ++i)
    {
        auto* p = params_[static_cast<std::size_t>(i)];
        const float after = p->getValue();
        const float before = beforeNormalised[static_cast<std::size_t>(i)];
        if (before != after)
        {
            e.changes[static_cast<std::size_t>(e.count)] = { i, before, after };
            if (e.firstParam < 0)
                e.firstParam = i;
            ++e.count;
        }
    }
    if (e.count > 0)
        commit_(e, name);
}

void EditHistory::clear()
{
    cursor_ = 0;
    count_ = 0;
    undoCount_ = 0;
    pending_.count = 0;
    pending_.firstParam = -1;
    openCount_ = 0;
    notifyChanged_();
}

void EditHistory::parameterGestureChanged(const int index, const bool starting)
{
    JUCE_ASSERT_MESSAGE_THREAD

    // Applying an undo or redo records nothing.
    if (applying_)
        return;

    if (starting)
    {
        // Record the before value once per parameter.
        bool found = false;
        for (int i = 0; i < pending_.count; ++i)
            if (pending_.changes[static_cast<std::size_t>(i)].paramIndex == index)
            {
                found = true;
                break;
            }
        if (! found && index < static_cast<int>(params_.size()))
        {
            pending_.changes[static_cast<std::size_t>(pending_.count)] = {
                index, params_[static_cast<std::size_t>(index)]->getValue(), 0.0f };
            if (pending_.firstParam < 0)
                pending_.firstParam = index;
            ++pending_.count;
        }
        ++openCount_;
    }
    else
    {
        // Record the after value.
        if (index < static_cast<int>(params_.size()))
        {
            bool found = false;
            for (int i = 0; i < pending_.count; ++i)
                if (pending_.changes[static_cast<std::size_t>(i)].paramIndex == index)
                {
                    pending_.changes[static_cast<std::size_t>(i)].after =
                        params_[static_cast<std::size_t>(index)]->getValue();
                    found = true;
                    break;
                }
            if (! found)
            {
                const float v = params_[static_cast<std::size_t>(index)]->getValue();
                pending_.changes[static_cast<std::size_t>(pending_.count)] = { index, v, v };
                if (pending_.firstParam < 0)
                    pending_.firstParam = index;
                ++pending_.count;
            }
        }
        --openCount_;
        if (openCount_ <= 0)
        {
            openCount_ = 0;
            // Drop changes where before == after.
            Entry e;
            for (int i = 0; i < pending_.count; ++i)
            {
                const auto& c = pending_.changes[static_cast<std::size_t>(i)];
                if (c.before != c.after)
                {
                    e.changes[static_cast<std::size_t>(e.count)] = c;
                    if (e.firstParam < 0)
                        e.firstParam = c.paramIndex;
                    ++e.count;
                }
            }
            if (e.count > 0)
                commit_(e, entryName_(e));
            pending_.count = 0;
            pending_.firstParam = -1;
        }
    }
}

void EditHistory::commit_(const Entry& e, const String& name)
{
    // The ring write. Truncate the redo tail and evict past capacity.
    entries_[static_cast<std::size_t>(cursor_)] = e;
    cursor_ = (cursor_ + 1) % kCapacity;
    ++undoCount_;
    count_ = undoCount_; // truncate redo
    if (count_ > kCapacity)
    {
        undoCount_ = kCapacity;
        count_ = kCapacity;
    }
    notifyChanged_();
}

void EditHistory::notifyChanged_()
{
    if (onChanged)
        onChanged();
}

String EditHistory::entryName_(const Entry& e) const
{
    if (e.count <= 0 || e.firstParam < 0)
        return {};
    if (e.count == 1)
        return params_[static_cast<std::size_t>(e.firstParam)]->getName(32);
    return params_[static_cast<std::size_t>(e.firstParam)]->getName(32) + " +" + String(e.count);
}

} // namespace MarsDSP::State
