#pragma once

#ifndef CHRONOS_PRESET_STORE_H
#define CHRONOS_PRESET_STORE_H

#include <JuceHeader.h>

namespace MarsDSP::Presets {

// The preset file extension.
inline constexpr const char* kPresetExtension = ".chronos";

// Metadata property names stored on the preset state tree.
// An absent property reads as its default, so an old session
// and an old preset both load unchanged. A new preset loads
// in an older build with the extra properties ignored.
inline constexpr const char* kPresetNameProp     = "presetName";
inline constexpr const char* kPresetAuthorProp   = "presetAuthor";
inline constexpr const char* kPresetCategoryProp = "presetCategory";

// A preset entry in the directory listing.
struct PresetEntry {
    String name;
    String bank;
    bool factory = false;
};

// The file layer. Enumerates, reads, writes, renames, and
// deletes preset files. Knows nothing about the processor.
class PresetStore {
public:
    PresetStore() = default;

    // The root preset directory.
    File getRootDirectory() const;

    // Override the root directory for testing.
    void setRootDirectory(const File& dir) { rootDir_ = dir; }

    // Ensure the root directory exists. Create it on first use.
    // Return false when the directory is absent and creation fails.
    bool ensureRootDirectory();

    // Enumerate the user presets under the root directory.
    // One level of subdirectory is a bank; deeper nesting is ignored.
    std::vector<PresetEntry> enumerateUserPresets() const;

    // Read a preset file and parse it as XML.
    // Return nullptr on a parse failure or a missing file.
    static std::unique_ptr<XmlElement> loadPresetFile(const File& file);

    // Write a preset XML string to a file.
    // Return false on a write failure.
    static bool savePresetFile(const File& file, const String& xmlText, bool allowOverwrite = false);

    // Delete a preset file. Return false when the file is absent.
    static bool deletePresetFile(const File& file);

    // Rename a preset file. Refuse to overwrite an existing file.
    static bool renamePresetFile(const File& file, const String& newName);

    // The file path for a bank and name.
    File presetFile(const String& bank, const String& name) const;

    // The bank a file belongs to, from its position under the root.
    // The parent directory name sits one level down. An empty string
    // when the file sits directly in the root.
    String bankForFile(const File& file) const;

    // Sanitise a name for the file system.
    // Strip path separators, reserved characters, and leading dots.
    // Return an empty string for a reserved device name or when
    // nothing survives.
    static String sanitiseName(const String& name);

private:
    File rootDir_;
};

} // namespace MarsDSP::Presets

#endif
