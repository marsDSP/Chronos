#include "Card.h"
#include "Fonts.h"
#include "MetricsConsumer.h"

namespace MarsDSP::GUI {

Card::Page& Card::pageAt_(int index) noexcept
{
    return pages_[static_cast<std::size_t>(index)];
}

void Card::addPage(const String& title, std::unique_ptr<Component> panel)
{
    Page page;
    page.title = title.toUpperCase();
    page.panel = std::move(panel);

    auto tab = std::make_unique<TabButton>(page.title);
    tab->setMetrics(metrics_);
    tab->setAccentColour(accent_);
    tab->setTooltip("Show the " + page.title + " controls.");

    // A click shows the page and reports it. A click on the active tab
    // does nothing.
    const int index = static_cast<int>(pages_.size());
    tab->onClick = [this, index]
    {
        if (selectedPage_ == index)
            return;
        setSelectedPage(index);
        if (onPageChanged != nullptr)
            onPageChanged(index);
    };

    page.tab = std::move(tab);
    addAndMakeVisible(page.panel.get());
    addAndMakeVisible(page.tab.get());

    if (auto* ac = dynamic_cast<AccentConsumer*>(page.panel.get()))
        ac->setAccentColour(accent_);

    pages_.push_back(std::move(page));
    setSelectedPage(selectedPage_);
    resized();
    repaint();
}

void Card::setSelectedPage(int index)
{
    const int count = static_cast<int>(pages_.size());
    if (count == 0)
        return;

    selectedPage_ = std::clamp(index, 0, count - 1);

    // Exactly one page is visible. Every page keeps its bounds.
    for (int i = 0; i < count; ++i)
    {
        auto& page = pageAt_(i);
        const bool active = (i == selectedPage_);
        page.panel->setVisible(active);
        if (page.tab != nullptr)
            page.tab->setActive(active);
    }

    resized();
    repaint();
}

void Card::setPageMark(int index, bool marked)
{
    if (index < 0 || index >= static_cast<int>(pages_.size()))
        return;

    if (auto* tab = pageAt_(index).tab.get())
        tab->setMarked(marked);
}

void Card::setAccentColour(const Colour c)
{
    accent_ = c;

    for (auto& page : pages_)
    {
        if (auto* ac = dynamic_cast<AccentConsumer*>(page.panel.get()))
            ac->setAccentColour(c);
        if (page.tab != nullptr)
            page.tab->setAccentColour(c);
    }

    repaint();
}

void Card::setMetrics(const Metrics& m)
{
    metrics_ = m;

    for (auto& page : pages_)
    {
        if (auto* mc = dynamic_cast<MetricsConsumer*>(page.panel.get()))
            mc->setMetrics(m);
        if (page.tab != nullptr)
            page.tab->setMetrics(m);
    }

    resized();
    repaint();
}

void Card::setEnablement(const EnablementState& state)
{
    for (auto& page : pages_)
        if (auto* ec = dynamic_cast<EnablementConsumer*>(page.panel.get()))
            ec->setControlsEnabled(state);
}

void Card::paint(Graphics& g)
{
    const auto bounds = getLocalBounds().toFloat();
    const float r = metrics_.pxf(static_cast<float>(Metrics::kCardCornerRadius));
    const float sw = metrics_.stroke(static_cast<float>(Metrics::kCardBorderStroke));

    g.setColour(Colours::panelBackground);
    g.fillRoundedRectangle(bounds, r);

    g.setColour(tint(Colours::panelBackground, accent_, kTintCardBorder));
    g.drawRoundedRectangle(bounds.reduced(sw / 2), r, sw);

    // The single page keeps the painted title. The tabs carry the titles
    // of a paged card, so the card paints no text of its own then.
    if (pages_.size() < 2 && ! pages_.empty())
    {
        const int titleH = metrics_.px(static_cast<float>(Metrics::kCardTitleH));
        const int hpad = metrics_.px(static_cast<float>(Metrics::kCardHPad));
        Font titleFont = Fonts::font(Fonts::Weight::Semibold, metrics_.font(Metrics::kCardTitleFont));
        titleFont.setExtraKerningFactor(Metrics::kTitleTracking);
        g.setFont(titleFont);
        g.setColour(Colours::textPrimary);
        g.drawText(pageAt_(0).title, hpad, 0, getWidth() - 2 * hpad, titleH,
                   Justification::centredLeft, false);
    }
}

void Card::resized()
{
    const int border = metrics_.px(static_cast<float>(Metrics::kCardBorderStroke));
    const int hpad = metrics_.px(static_cast<float>(Metrics::kCardHPad));
    const int titleH = metrics_.px(static_cast<float>(Metrics::kCardTitleH));
    const int titleGap = metrics_.px(static_cast<float>(Metrics::kCardTitleGap));
    const int bottomPad = metrics_.px(static_cast<float>(Metrics::kCardBottomPad));

    const int x = border + hpad;
    const int y = border + titleH + titleGap;
    const int w = getWidth() - 2 * (border + hpad);
    const int h = getHeight() - y - bottomPad;

    // Every page holds the same content rectangle, hidden or not. A page
    // switch therefore never relayouts the page it shows.
    for (auto& page : pages_)
        if (page.panel != nullptr)
            page.panel->setBounds(x, y, w, h);

    // The tab row. The tabs run left to right from the side pad.
    const int count = static_cast<int>(pages_.size());
    if (count >= 2)
    {
        const int gap = metrics_.px(static_cast<float>(Metrics::kTabGap));
        int tx = hpad;
        for (auto& page : pages_)
        {
            const int tw = page.tab->preferredWidthPx(metrics_);
            page.tab->setBounds(tx, 0, tw, titleH);
            tx += tw + gap;
        }
    }
}

} // namespace MarsDSP::GUI
