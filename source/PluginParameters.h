// Central parameter-ID and plugin-state version contract.
//
// Rules of the contract:
//   1. Parameter IDs are referenced ONLY through the constants in this file.
//      Never hard-code the string literal elsewhere. Renaming a param means
//      editing this file (and bumping kPluginStateVersion + adding a migration).
//   2. Every saved state includes the property "pluginStateVersion" at the
//      root of the APVTS ValueTree. On load, the processor dispatches the old
//      tree through migrateStateTree() before calling replaceState().
//   3. Adding a brand-new parameter with a sensible default is SAFE without a
//      version bump: JUCE's APVTS leaves missing parameters at their default
//      when loading older states.
//   4. Any of the following REQUIRE a version bump AND a migration branch:
//        - renaming / removing a parameter
//        - changing a parameter's units, range, skew, or default in a way
//          that changes interpretation of a stored normalized value
//        - splitting or merging parameters
#pragma once
#include <JuceHeader.h>

namespace Chronos
{
    // --------------------------------------------------------------
    // Version
    // --------------------------------------------------------------
    // Bump whenever a migration is required. Keep in lockstep with
    // MarsDSP::DSP::DelayEngine<float>::streamingVersion when the engine's
    // schema changes.
    inline constexpr int kPluginStateVersion = 1;

    // --------------------------------------------------------------
    // Parameter IDs (stable across the plugin's lifetime)
    // --------------------------------------------------------------
    namespace ParamID
    {
        inline constexpr auto kDelayTime = "delayTime";
        inline constexpr auto kMix       = "mix";
        inline constexpr auto kFeedback  = "feedback";
        inline constexpr auto kLowCut    = "lowCut";
        inline constexpr auto kHighCut   = "highCut";
        inline constexpr auto kCrossfeed = "crossfeed";
        inline constexpr auto kMono      = "mono";
        inline constexpr auto kBypass    = "bypass";
    } // namespace ParamID

    // --------------------------------------------------------------
    // State-tree version property name
    // --------------------------------------------------------------
    inline constexpr auto kStateVersionProperty = "pluginStateVersion";

    // --------------------------------------------------------------
    // Migrate an older APVTS ValueTree to the current schema.
    //
    // This is best-effort: for any version gap we know about, rewrite the
    // tree so that (a) parameters carry their current ID, and (b) values are
    // already in the current unit/range space. Anything we can't resolve is
    // left alone; APVTS::replaceState() will then fall back to defaults for
    // missing parameters.
    //
    // `fromVersion` is whatever was stored in kStateVersionProperty (0 when
    // the tree was saved by a pre-versioning build).
    // --------------------------------------------------------------
    inline void migrateStateTree(juce::ValueTree& tree, int fromVersion)
    {
        if (fromVersion >= kPluginStateVersion) return;

        // === v0 -> v1 ===
        // v0 was the pre-versioning build. It shipped with only:
        //   delayTime, mix, feedback, mono, bypass.
        // v1 added: lowCut, highCut, crossfeed. These are handled automatically
        // by APVTS (missing params stay at default), so no tree surgery needed.
        if (fromVersion < 1)
        {
            // No-op. Example for the future:
            // renameParam(tree, "oldName", ParamID::kSomething);
        }

        tree.setProperty(kStateVersionProperty, kPluginStateVersion, nullptr);
    }

    // --------------------------------------------------------------
    // Helper for future migrations: rewrite every PARAM child whose `id`
    // matches oldId to newId, preserving the value.
    // --------------------------------------------------------------
    inline void renameParam(juce::ValueTree& tree,
                            const juce::Identifier& idProp,
                            juce::StringRef oldId,
                            juce::StringRef newId)
    {
        const juce::String newIdStr (newId);
        for (int i = 0; i < tree.getNumChildren(); ++i)
        {
            auto child = tree.getChild(i);
            if (child.hasProperty(idProp)
                && child.getProperty(idProp).toString() == oldId)
            {
                child.setProperty(idProp, juce::var(newIdStr), nullptr);
            }
        }
    }

} // namespace Chronos
