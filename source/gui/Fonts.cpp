#include "Fonts.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#if __has_include("ChronosFonts.h")
#  include "ChronosFonts.h"
#  define CHRONOS_HAS_FONT_RESOURCE 1
#else
#  define CHRONOS_HAS_FONT_RESOURCE 0
#endif

namespace MarsDSP::GUI::Fonts {
namespace {

// Hold one typeface pointer per weight. Build the cache once.
struct TypefaceCache {
    Typeface::Ptr regular;
    Typeface::Ptr medium;
    Typeface::Ptr semibold;

    TypefaceCache()
    {
#if CHRONOS_HAS_FONT_RESOURCE
        regular  = load("ClashGroteskRegular_ttf");
        medium   = load("ClashGroteskMedium_ttf");
        semibold = load("ClashGroteskSemibold_ttf");
#endif
    }

#if CHRONOS_HAS_FONT_RESOURCE
    static Typeface::Ptr load(const char* resourceName)
    {
        int size = 0;
        const char* data = ChronosFonts::getNamedResource(resourceName, size);
        if (data == nullptr || size <= 0)
            return {};
        return Typeface::createSystemTypefaceFor(data, static_cast<size_t>(size));
    }
#endif
};

const TypefaceCache& cache()
{
    static const TypefaceCache c;
    return c;
}

// The default system face. Use it when the resource is absent or a weight fails to load.
Typeface::Ptr fallbackTypeface()
{
    return Font{}.getTypefacePtr();
}

} // namespace

Typeface::Ptr typefaceFor(const Weight weight)
{
    const auto& c = cache();

    switch (weight)
    {
        case Weight::Medium:   return c.medium   != nullptr ? c.medium   : fallbackTypeface();
        case Weight::Semibold: return c.semibold != nullptr ? c.semibold : fallbackTypeface();
        case Weight::Regular:
        default:               return c.regular  != nullptr ? c.regular  : fallbackTypeface();
    }
}

Font font(const Weight weight, const float height)
{
    return Font { FontOptions { typefaceFor(weight) }.withHeight(jmax(1.0f, height)) };
}

String shortLabel(const String& full)
{
    static const std::pair<String, String> table[] = {
        { "LEFT TIME",      "LEFT"  },
        { "RIGHT TIME",     "RIGHT" },
        { "FEEDBACK",       "FDBK"  },
        { "CROSSFEED",      "CROSS" },
        { "LOOP DRIVE",     "DRIVE" },
        { "OUTPUT HPF",     "HPF"   },
        { "OUTPUT LPF",     "LPF"   },
        { "LOOP LPF",       "LPF"   },
        { "DIFFUSION",      "DIFF"  },
        { "DIFFUSER SIZE",  "SIZE"  },
        { "MOD DEPTH",      "DEPTH" },
        { "MOD RATE",       "RATE"  },
        { "ANTI-ALIAS",     "ALIAS" },
    };

    for (const auto& [f, s] : table)
        if (full.equalsIgnoreCase(f))
            return s;

    return full;
}

float digitAdvance(const Font& font)
{
    // Cache one advance per font height. The widest digit sets the advance.
    struct Entry { float height; float advance; };
    static std::vector<Entry> entries;

    const float h = font.getHeight();

    for (const auto& e : entries)
        if (std::fabs(e.height - h) < 0.01f)
            return e.advance;

    float widest = 0.0f;
    for (char d = '0'; d <= '9'; ++d)
        widest = std::max(widest, font.getStringWidthFloat(String::charToString(static_cast<juce_wchar>(d))));

    // Hold a positive advance so an empty metric never yields zero.
    widest = std::max(widest, 1.0f);
    entries.push_back({ h, widest });
    return widest;
}

void drawFixedAdvanceText(Graphics& g, const Font& font, const String& text,
                          const Rectangle<float>& area, const Colour colour)
{
    const float adv = digitAdvance(font);
    const float baselineY = area.getCentreY() + (font.getAscent() - font.getDescent()) * 0.5f;

    float totalW = 0.0f;
    for (const auto c : text)
        totalW += (c >= '0' && c <= '9') ? adv : font.getStringWidthFloat(String::charToString(c));

    float x = area.getCentreX() - totalW * 0.5f;
    g.setFont(font);
    g.setColour(colour);

    for (const auto c : text)
    {
        const String ch = String::charToString(c);

        if (c >= '0' && c <= '9')
        {
            g.drawSingleLineText(ch, roundToInt(x + adv * 0.5f), roundToInt(baselineY),
                                  Justification::horizontallyCentred);
            x += adv;
        }
        else
        {
            g.drawSingleLineText(ch, roundToInt(x), roundToInt(baselineY), Justification::left);
            x += font.getStringWidthFloat(ch);
        }
    }
}

} // namespace MarsDSP::GUI::Fonts
