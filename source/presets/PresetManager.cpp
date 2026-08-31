#include "PresetManager.h"
#include "../ChronosProcessor.h"
#include "../ChronosParameters.h"

namespace MarsDSP::Presets {

// The 28 APVTS parameter IDs. Register one dirty listener per ID.
static const ParameterID kParamIDs[] = {
    gainParamID,          bitsParamID,         delayTimeParamID,    delayTimeRParamID,
    timeLinkParamID,       delaySyncParamID,    delayDivisionParamID, delayModeParamID,
    bypassParamID,         filterModeParamID,   hpfFreqParamID,      lpfFreqParamID,
    mixParamID,            driveParamID,        adaaOrderParamID,     feedbackParamID,
    dampHzParamID,         loopCutHzParamID,    crossFeedParamID,     loopDriveParamID,
    loopSatOrderParamID,   delayModDepthParamID, delayModRateHzParamID, enableDiffuserParamID,
    diffusionParamID,      diffuserSizeParamID,  diffModDepthParamID,  diffModRateHzParamID
};

PresetManager::PresetManager(AudioProcessor& proc, AudioProcessorValueTreeState& apvts)
    : processorRef_(proc), apvtsRef_(apvts)
{
    registerParameterListeners_();
}

PresetManager::PresetManager(ChronosProcessor& proc)
    : processorRef_(proc), apvtsRef_(proc.getAPVTS())
{
    registerParameterListeners_();
}

PresetManager::~PresetManager()
{
    unregisterParameterListeners_();
}

// One atomic store and nothing else. Runs on the audio thread.
void PresetManager::parameterChanged(const String&, float)
{
    modified_.store(true, std::memory_order_relaxed);
}

void PresetManager::registerParameterListeners_()
{
    for (const auto& pid : kParamIDs)
        apvtsRef_.addParameterListener(pid.getParamID(), this);
}

void PresetManager::unregisterParameterListeners_()
{
    for (const auto& pid : kParamIDs)
        apvtsRef_.removeParameterListener(pid.getParamID(), this);
}

bool PresetManager::saveAs(const String& name, const String& author, const String& category)
{
    const auto cleanName = PresetStore::sanitiseName(name);
    if (cleanName.isEmpty()) return false;

    MemoryBlock block;
    processorRef_.getStateInformation(block);

    auto xml = AudioProcessor::getXmlFromBinary(block.getData(), (int) block.getSize());
    if (xml == nullptr) return false;

    auto state = ValueTree::fromXml(*xml);
    state.setProperty(kPresetNameProp, cleanName, nullptr);
    state.setProperty(kPresetAuthorProp, author, nullptr);
    state.setProperty(kPresetCategoryProp, category, nullptr);

    const auto file = store_.presetFile(presetBank_, cleanName);
    if (! store_.savePresetFile(file, state.createXml()->toString(), false))
        return false;

    presetName_     = cleanName;
    presetAuthor_   = author;
    presetCategory_ = category;
    currentFile_    = file;
    isFactory_      = false;
    modified_.store(false, std::memory_order_relaxed);
    return true;
}

bool PresetManager::saveCurrent(const String& author, const String& category)
{
    if (isFactory_ || currentFile_ == File() || presetName_.isEmpty()) return false;

    MemoryBlock block;
    processorRef_.getStateInformation(block);

    auto xml = AudioProcessor::getXmlFromBinary(block.getData(), (int) block.getSize());
    if (xml == nullptr) return false;

    auto state = ValueTree::fromXml(*xml);
    state.setProperty(kPresetNameProp, presetName_, nullptr);
    state.setProperty(kPresetAuthorProp, author, nullptr);
    state.setProperty(kPresetCategoryProp, category, nullptr);

    if (! store_.savePresetFile(currentFile_, state.createXml()->toString(), true))
        return false;

    presetAuthor_   = author;
    presetCategory_ = category;
    modified_.store(false, std::memory_order_relaxed);
    return true;
}

// Apply a state tree through the processor recall path.
// Return false on a root tag mismatch.
bool PresetManager::applyStateXml_(const XmlElement& xml)
{
    if (! xml.hasTagName(apvtsRef_.state.getType()))
        return false;

    MemoryBlock blob;
    AudioProcessor::copyXmlToBinary(xml, blob);

    processorRef_.setStateInformation(blob.getData(), static_cast<int>(blob.getSize()));
    return true;
}

bool PresetManager::loadPreset(const File& file)
{
    auto xml = PresetStore::loadPresetFile(file);
    if (xml == nullptr) return false;

    if (! applyStateXml_(*xml))
        return false;

    loadIdentity(file);

    // replaceState fires the parameter listeners that would set the
    // flag immediately. Defer the clear so it runs after them.
    MessageManager::callAsync([this]
    {
        modified_.store(false, std::memory_order_relaxed);
    });

    return true;
}

void PresetManager::loadIdentity(const File& file)
{
    auto xml = PresetStore::loadPresetFile(file);
    if (xml == nullptr) return;

    presetName_     = xml->getStringAttribute(kPresetNameProp);
    presetAuthor_   = xml->getStringAttribute(kPresetAuthorProp);
    presetCategory_ = xml->getStringAttribute(kPresetCategoryProp);
    currentFile_    = file;
    isFactory_      = false;
}

bool PresetManager::deleteCurrent()
{
    if (isFactory_ || currentFile_ == File()) return false;
    if (! store_.deletePresetFile(currentFile_)) return false;
    presetName_.clear();
    presetBank_.clear();
    currentFile_ = File();
    return true;
}

bool PresetManager::renameCurrent(const String& newName)
{
    if (isFactory_ || currentFile_ == File()) return false;
    if (! store_.renamePresetFile(currentFile_, newName)) return false;
    presetName_ = PresetStore::sanitiseName(newName);
    currentFile_ = currentFile_.getSiblingFile(presetName_ + String(kPresetExtension));
    return true;
}

String PresetManager::copyPresetXml()
{
    MemoryBlock block;
    processorRef_.getStateInformation(block);

    auto xml = AudioProcessor::getXmlFromBinary(block.getData(), static_cast<int>(block.getSize()));
    if (xml == nullptr) return {};

    return xml->toString();
}

bool PresetManager::pastePresetXml(const String& xmlText)
{
    auto xml = parseXML(xmlText);
    if (xml == nullptr) return false;

    if (! applyStateXml_(*xml))
        return false;

    // An unnamed modified patch. The host owns the name.
    presetName_.clear();
    presetBank_.clear();
    currentFile_ = File();
    isFactory_ = false;
    modified_.store(true, std::memory_order_relaxed);
    return true;
}

} // namespace MarsDSP::Presets
