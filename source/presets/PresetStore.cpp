#include "PresetStore.h"

namespace MarsDSP::Presets {

File PresetStore::getRootDirectory() const
{
    if (rootDir_ != File())
        return rootDir_;
    return File::getSpecialLocation(File::userApplicationDataDirectory)
        .getChildFile("MarsDSP").getChildFile("Chronos").getChildFile("Presets");
}

bool PresetStore::ensureRootDirectory()
{
    const auto root = getRootDirectory();
    if (root.isDirectory()) return true;
    return root.createDirectory().wasOk();
}

std::vector<PresetEntry> PresetStore::enumerateUserPresets() const
{
    std::vector<PresetEntry> entries;
    const auto root = getRootDirectory();
    if (! root.isDirectory()) return entries;

    // One level of subdirectory is a bank.
    for (const auto& bank : root.findChildFiles(File::findDirectories, false))
    {
        for (const auto& file : bank.findChildFiles(File::findFiles, false, "*" + String(kPresetExtension)))
        {
            PresetEntry e;
            e.bank = bank.getFileName();
            e.name = file.getFileNameWithoutExtension();
            e.factory = false;
            entries.push_back(std::move(e));
        }
    }

    // Presets directly under the root carry an empty bank.
    for (const auto& file : root.findChildFiles(File::findFiles, false, "*" + String(kPresetExtension)))
    {
        PresetEntry e;
        e.bank = {};
        e.name = file.getFileNameWithoutExtension();
        e.factory = false;
        entries.push_back(std::move(e));
    }

    return entries;
}

std::unique_ptr<XmlElement> PresetStore::loadPresetFile(const File& file)
{
    if (! file.existsAsFile()) return {};
    return parseXML(file.loadFileAsString());
}

bool PresetStore::savePresetFile(const File& file, const String& xmlText, bool allowOverwrite)
{
    if (! allowOverwrite && file.existsAsFile()) return false;
    if (! file.getParentDirectory().createDirectory().wasOk()) return false;
    return file.replaceWithText(xmlText);
}

bool PresetStore::deletePresetFile(const File& file)
{
    if (! file.existsAsFile()) return false;
    return file.deleteFile();
}

bool PresetStore::renamePresetFile(const File& file, const String& newName)
{
    if (! file.existsAsFile()) return false;
    const auto target = file.getSiblingFile(sanitiseName(newName) + String(kPresetExtension));
    if (target == file) return true;
    // Refuse to overwrite an existing file without an explicit replace.
    if (target.existsAsFile()) return false;
    return file.moveFileTo(target);
}

File PresetStore::presetFile(const String& bank, const String& name) const
{
    const auto root = getRootDirectory();
    const auto fileName = sanitiseName(name) + String(kPresetExtension);
    if (bank.isEmpty()) return root.getChildFile(fileName);
    return root.getChildFile(bank).getChildFile(fileName);
}

String PresetStore::sanitiseName(const String& name)
{
    auto trimmed = name.trim();

    // Strip path separators.
    for (auto i = 0; i < trimmed.length(); ++i)
    {
        const auto c = trimmed[i];
        if (c == '/' || c == '\\')
            trimmed = trimmed.replaceSection(i, 1, "_");
    }

    // Strip leading dots.
    while (trimmed.startsWithChar('.'))
        trimmed = trimmed.substring(1);

    return trimmed;
}

} // namespace MarsDSP::Presets
