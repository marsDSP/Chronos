#pragma once

#ifndef CHRONOS_HINT_ROUTER_H
#define CHRONOS_HINT_ROUTER_H

#include <JuceHeader.h>
#include <functional>
#include <utility>

namespace MarsDSP::GUI {

// Routes hover tooltips into a sink (the footer hint) instead of a popup
// window. The editor registers it as a mouse listener for all nested
// children; on enter and move it walks up from the hovered component to
// the nearest TooltipClient and publishes its tip, and an exit clears it.
// Every control keeps its setTooltip string; only the presentation moves.
class HintRouter : public MouseListener {
public:
    explicit HintRouter(std::function<void(const String&)> sink)
        : sink_(std::move(sink)) {}

    void mouseEnter(const MouseEvent& e) override { publish_(e.eventComponent); }
    void mouseMove(const MouseEvent& e) override  { publish_(e.eventComponent); }
    void mouseExit(const MouseEvent&) override    { clear_(); }

private:
    void publish_(Component* hovered)
    {
        for (auto* c = hovered; c != nullptr; c = c->getParentComponent())
        {
            if (auto* client = dynamic_cast<TooltipClient*>(c))
            {
                if (const String tip = client->getTooltip(); tip.isNotEmpty())
                {
                    if (tip != last_)
                    {
                        last_ = tip;
                        sink_(tip);
                    }
                    return;
                }
            }
        }
        clear_();
    }

    void clear_()
    {
        if (last_.isEmpty())
            return;
        last_.clear();
        sink_({});
    }

    std::function<void(const String&)> sink_;
    String last_;
};

} // namespace MarsDSP::GUI

#endif
