#include "PresetBar.h"
#include "../presets/PresetManager.h"
#include "../presets/PresetStore.h"
#include "Fonts.h"

#include <map>

namespace MarsDSP::GUI {

// Menu item IDs.
enum MenuCommand {
    kMenuSave = 1,
    kMenuSaveAs,
    kMenuRename,
    kMenuDelete,
    kMenuCopy,
    kMenuPaste,
    kMenuImport,
    kMenuExport,
    kMenuShowFolder,
    kMenuFactoryPresetStart = 500,
    kMenuUserPresetStart = 1000
};

PresetBar::PresetBar(MarsDSP::Presets::PresetManager& pm)
    : pm_(pm)
{
    setTitle("Preset Bar");
    setHelpText("Browse, save, and load presets.");
    refreshName_();
    modifiedSeen_ = pm_.isModified();
    startTimerHz(10);
}

PresetBar::~PresetBar()
{
    stopTimer();
}

void PresetBar::setMetrics(const Metrics& m)
{
    metrics_ = m;
    resized();
    repaint();
}

void PresetBar::setAccentColour(Colour c)
{
    accent_ = c;
    repaint();
}

void PresetBar::refreshName_()
{
    displayedName_ = pm_.getCurrentName();
}

void PresetBar::timerCallback()
{
    const bool modified = pm_.isModified();
    const String name = pm_.getCurrentName();
    if (modified != modifiedSeen_ || name != displayedName_)
    {
        modifiedSeen_ = modified;
        displayedName_ = name;
        repaint();
    }
}

void PresetBar::paint(Graphics& g)
{
    const auto& m = metrics_;
    const float sw = m.stroke(Metrics::kHairline);
    const float corner = m.pxf(Metrics::kCornerSmall);
    const auto bounds = getLocalBounds().toFloat();

    g.setColour(Colours::headerBackground);
    g.fillRoundedRectangle(bounds, corner);

    g.setColour(tint(Colours::headerBackground, accent_, kTintCardBorder));
    g.drawRoundedRectangle(bounds.reduced(sw * 0.5f), corner, sw);

    auto drawChevron = [&](const Rectangle<int>& area, bool left, bool hover)
    {
        const float cx = static_cast<float>(area.getCentreX());
        const float cy = static_cast<float>(area.getCentreY());
        const float sz = static_cast<float>(area.getWidth()) * 0.22f;
        Path p;
        if (left)
        {
            p.startNewSubPath(cx + sz, cy - sz);
            p.lineTo(cx - sz, cy);
            p.lineTo(cx + sz, cy + sz);
        }
        else
        {
            p.startNewSubPath(cx - sz, cy - sz);
            p.lineTo(cx + sz, cy);
            p.lineTo(cx - sz, cy + sz);
        }
        g.setColour(hover ? Colours::textPrimary : Colours::textMuted);
        g.strokePath(p, PathStrokeType(m.stroke(Metrics::kIconStroke),
                    PathStrokeType::curved, PathStrokeType::rounded));
    };
    drawChevron(prevArea_, true, hoveredRegion_ == 1);
    drawChevron(nextArea_, false, hoveredRegion_ == 3);

    // Menu glyph: three horizontal lines.
    {
        const float cx = static_cast<float>(menuArea_.getCentreX());
        const float cy = static_cast<float>(menuArea_.getCentreY());
        const float lineW = static_cast<float>(menuArea_.getWidth()) * 0.4f;
        const float gap = static_cast<float>(menuArea_.getHeight()) * 0.14f;
        const float lineSw = m.stroke(Metrics::kIconStroke);
        g.setColour(hoveredRegion_ == 4 ? Colours::textPrimary : Colours::textMuted);
        for (int i = -1; i <= 1; ++i)
            g.drawLine(cx - lineW * 0.5f, cy + static_cast<float>(i) * gap,
                       cx + lineW * 0.5f, cy + static_cast<float>(i) * gap, lineSw);
    }

    // Preset name.
    {
        const String name = displayedName_.isEmpty() ? String("Init") : displayedName_;
        const String text = modifiedSeen_ ? (name + " *") : name;
        g.setFont(Fonts::font(Fonts::Weight::Regular, m.font(Metrics::kPresetBarFont)));
        g.setColour(modifiedSeen_ ? accent_ : Colours::textBright);
        g.drawText(text, nameArea_, Justification::centred, true);
    }
}

void PresetBar::resized()
{
    const auto& m = metrics_;
    const int h = getHeight();
    const int arrow = m.px(static_cast<float>(Metrics::kPresetBarArrow));
    const int menu = m.px(static_cast<float>(Metrics::kPresetBarMenu));

    prevArea_ = Rectangle<int>(0, (h - arrow) / 2, arrow, arrow);
    menuArea_ = Rectangle<int>(getWidth() - menu, (h - menu) / 2, menu, menu);
    nextArea_ = Rectangle<int>(menuArea_.getX() - arrow, (h - arrow) / 2, arrow, arrow);
    nameArea_ = Rectangle<int>(prevArea_.getRight(), 0,
                               nextArea_.getX() - prevArea_.getRight(), h);
}

void PresetBar::mouseMove(const MouseEvent& e)
{
    const auto pos = e.position.toInt();
    int region = 0;
    if (prevArea_.contains(pos)) region = 1;
    else if (nameArea_.contains(pos)) region = 2;
    else if (nextArea_.contains(pos)) region = 3;
    else if (menuArea_.contains(pos)) region = 4;
    if (region != hoveredRegion_)
    {
        hoveredRegion_ = region;
        repaint();
    }
}

void PresetBar::mouseExit(const MouseEvent&)
{
    if (hoveredRegion_ != 0)
    {
        hoveredRegion_ = 0;
        repaint();
    }
}

void PresetBar::mouseUp(const MouseEvent& e)
{
    if (! e.mouseWasClicked()) return;
    const auto pos = e.position.toInt();
    if (prevArea_.contains(pos)) stepPreset_(-1);
    else if (nextArea_.contains(pos)) stepPreset_(1);
    else if (nameArea_.contains(pos) || menuArea_.contains(pos)) showMenu_();
}

String PresetBar::getTooltip()
{
    const auto pos = getMouseXYRelative();
    if (prevArea_.contains(pos)) return "Select the previous preset.";
    if (nameArea_.contains(pos)) return "Click to open the preset menu.";
    if (nextArea_.contains(pos)) return "Select the next preset.";
    if (menuArea_.contains(pos)) return "Open the preset menu.";
    return {};
}

void PresetBar::stepPreset_(int direction)
{
    // Build the concatenated list: factory presets first, then user presets.
    auto presets = pm_.getFactoryPresets();
    auto user = pm_.getUserPresets();
    for (auto& e : user)
        presets.push_back(std::move(e));
    if (presets.empty()) return;

    int currentIdx = -1;
    for (int i = 0; i < static_cast<int>(presets.size()); ++i)
    {
        if (presets[static_cast<size_t>(i)].bank == pm_.getCurrentBank()
            && presets[static_cast<size_t>(i)].name == pm_.getCurrentName())
        {
            currentIdx = i;
            break;
        }
    }

    const int n = static_cast<int>(presets.size());
    int nextIdx = (currentIdx < 0) ? ((direction > 0) ? 0 : n - 1)
                                   : (currentIdx + direction + n) % n;

    const auto e = presets[static_cast<size_t>(nextIdx)];
    const auto safe = SafePointer<PresetBar>(this);
    confirmDiscardChanges_([safe, e]
    {
        if (safe == nullptr) return;
        if (e.factory)
        {
            if (! safe->pm_.loadFactoryPreset(e.name, e.bank))
                NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::WarningIcon,
                    "Preset Load Failed", "The factory preset could not load.");
        }
        else
        {
            const File file = safe->pm_.getStore().presetFile(e.bank, e.name);
            if (! safe->pm_.loadPreset(file))
                NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::WarningIcon,
                    "Preset Load Failed",
                    "The preset file could not load. " + safe->pm_.getLastError());
        }
        safe->refreshName_();
    });
}

// One guard for every load. When the current preset has unsaved
// changes, an async yes-no box names it and proceeds only on yes.
void PresetBar::confirmDiscardChanges_(std::function<void()> proceed)
{
    if (! pm_.isModified())
    {
        proceed();
        return;
    }

    const auto safe = SafePointer<PresetBar>(this);
    NativeMessageBox::showYesNoBox(MessageBoxIconType::WarningIcon,
        "Discard Changes?",
        "The preset \"" + pm_.getCurrentName() + "\" has unsaved changes. Discard them?",
        nullptr,
        ModalCallbackFunction::create([safe, proceed](int r) {
            if (safe != nullptr && r == 1)
                proceed();
        }));
}

void PresetBar::showMenu_()
{
    PopupMenu menu;
    const bool isFactory = pm_.isCurrentFactory();
    const bool hasCurrent = ! pm_.getCurrentName().isEmpty();

    menu.addItem(kMenuSave, "Save", ! isFactory && hasCurrent);
    menu.addItem(kMenuSaveAs, "Save As...", true);
    menu.addItem(kMenuRename, "Rename...", ! isFactory && hasCurrent);
    menu.addItem(kMenuDelete, "Delete", ! isFactory && hasCurrent);

    menu.addSeparator();
    menu.addItem(kMenuCopy, "Copy", true);
    menu.addItem(kMenuPaste, "Paste", true);

    menu.addSeparator();
    menu.addItem(kMenuImport, "Import...", true);
    menu.addItem(kMenuExport, "Export...", true);

    menu.addSeparator();
    menu.addItem(kMenuShowFolder, "Show Preset Folder", true);
    menu.addSeparator();

    // Preset list grouped by bank. Factory banks come first.
    menuPresetFiles_.clear();
    menuFactoryPresets_.clear();

    auto addBankSubmenu = [&](const String& bankName,
                              const std::vector<MarsDSP::Presets::PresetEntry>& entries,
                              bool factory)
    {
        if (entries.empty()) return;
        PopupMenu sub;
        for (const auto& e : entries)
        {
            const bool isCurrent = (e.bank == pm_.getCurrentBank()
                                    && e.name == pm_.getCurrentName());
            if (factory)
            {
                const int id = kMenuFactoryPresetStart
                    + static_cast<int>(menuFactoryPresets_.size());
                sub.addItem(id, e.name, true, isCurrent);
                menuFactoryPresets_.push_back(e);
            }
            else
            {
                const int id = kMenuUserPresetStart
                    + static_cast<int>(menuPresetFiles_.size());
                sub.addItem(id, e.name, true, isCurrent);
                menuPresetFiles_.push_back(pm_.getStore().presetFile(e.bank, e.name));
            }
        }
        menu.addSubMenu(bankName, sub, true);
    };

    // Factory presets grouped by bank, in the order they appear in the table.
    auto factoryPresets = pm_.getFactoryPresets();
    std::map<String, std::vector<MarsDSP::Presets::PresetEntry>> factoryByBank;
    for (const auto& e : factoryPresets)
        factoryByBank[e.bank].push_back(e);
    for (const auto& [bank, entries] : factoryByBank)
        addBankSubmenu(bank, entries, true);

    // User presets grouped by bank.
    auto userPresets = pm_.getUserPresets();
    std::map<String, std::vector<MarsDSP::Presets::PresetEntry>> userByBank;
    for (const auto& e : userPresets)
        userByBank[e.bank].push_back(e);
    for (const auto& [bank, entries] : userByBank)
    {
        const String bankName = bank.isEmpty() ? String("Presets") : bank;
        addBankSubmenu(bankName, entries, false);
    }

    const auto safe = SafePointer<PresetBar>(this);
    menu.showMenuAsync(PopupMenu::Options().withTargetComponent(this),
        [safe](int result) { if (safe != nullptr) safe->handleMenuResult_(result); });
}

void PresetBar::handleMenuResult_(int result)
{
    if (result == 0) return;

    if (result >= kMenuFactoryPresetStart && result < kMenuUserPresetStart)
    {
        const int idx = result - kMenuFactoryPresetStart;
        if (idx >= 0 && idx < static_cast<int>(menuFactoryPresets_.size()))
        {
            const auto e = menuFactoryPresets_[static_cast<size_t>(idx)];
            const auto safe = SafePointer<PresetBar>(this);
            confirmDiscardChanges_([safe, e]
            {
                if (safe == nullptr) return;
                if (! safe->pm_.loadFactoryPreset(e.name, e.bank))
                    NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::WarningIcon,
                        "Preset Load Failed", "The factory preset could not load.");
                safe->refreshName_();
            });
        }
        return;
    }

    if (result >= kMenuUserPresetStart)
    {
        const int idx = result - kMenuUserPresetStart;
        if (idx >= 0 && idx < static_cast<int>(menuPresetFiles_.size()))
        {
            const File file = menuPresetFiles_[static_cast<size_t>(idx)];
            const auto safe = SafePointer<PresetBar>(this);
            confirmDiscardChanges_([safe, file]
            {
                if (safe == nullptr) return;
                if (! safe->pm_.loadPreset(file))
                    NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::WarningIcon,
                        "Preset Load Failed",
                        "The preset file could not load. " + safe->pm_.getLastError());
                safe->refreshName_();
            });
        }
        return;
    }

    switch (result)
    {
        case kMenuSave:       doSave_(); break;
        case kMenuSaveAs:     doSaveAs_(); break;
        case kMenuRename:     doRename_(); break;
        case kMenuDelete:     doDelete_(); break;
        case kMenuCopy:       SystemClipboard::copyTextToClipboard(pm_.copyPresetXml()); break;
        case kMenuPaste:
        {
            const String xml = SystemClipboard::getTextFromClipboard();
            const auto safe = SafePointer<PresetBar>(this);
            confirmDiscardChanges_([safe, xml]
            {
                if (safe == nullptr) return;
                safe->pm_.pastePresetXml(xml);
                safe->refreshName_();
            });
            break;
        }
        case kMenuImport:     doImport_(); break;
        case kMenuExport:     doExport_(); break;
        case kMenuShowFolder: pm_.getStore().ensureRootDirectory();
                              pm_.getStore().getRootDirectory().revealToUser(); break;
        default: break;
    }
}

void PresetBar::doSave_()
{
    // Preserve the author and category from the current file.
    String author, category;
    const File file = pm_.getStore().presetFile(pm_.getCurrentBank(), pm_.getCurrentName());
    if (const auto xml = MarsDSP::Presets::PresetStore::loadPresetFile(file))
    {
        author = xml->getStringAttribute(MarsDSP::Presets::kPresetAuthorProp);
        category = xml->getStringAttribute(MarsDSP::Presets::kPresetCategoryProp);
    }
    if (! pm_.saveCurrent(author, category))
        NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::WarningIcon,
            "Save Failed", "The preset could not save.");
    refreshName_();
}

void PresetBar::doSaveAs_()
{
    auto* w = new AlertWindow("Save Preset", "Enter a name for the preset.", MessageBoxIconType::NoIcon);
    w->addTextEditor("name", pm_.getCurrentName(), "Name:");
    w->addButton("Save", 1, KeyPress(KeyPress::returnKey));
    w->addButton("Cancel", 0, KeyPress(KeyPress::escapeKey));

    const auto safe = SafePointer<PresetBar>(this);
    w->enterModalState(true, ModalCallbackFunction::create([safe, w](int r) {
        if (safe != nullptr && r == 1)
        {
            const String name = w->getTextEditorContents("name");
            safe->completeSaveAs_(name);
        }
    }), true);
}

void PresetBar::completeSaveAs_(const String& name)
{
    const String clean = MarsDSP::Presets::PresetStore::sanitiseName(name);
    if (clean.isEmpty())
    {
        NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::WarningIcon,
            "Invalid Name", "The preset name is empty. Enter a valid name.");
        return;
    }

    const File file = pm_.getStore().presetFile(pm_.getCurrentBank(), clean);
    if (file.existsAsFile())
    {
        const auto safe = SafePointer<PresetBar>(this);
        NativeMessageBox::showYesNoBox(MessageBoxIconType::WarningIcon,
            "Overwrite Preset", "A preset with that name exists. Overwrite it?",
            nullptr,
            ModalCallbackFunction::create([safe, clean](int r) {
                if (safe != nullptr && r == 1)
                {
                    // Keep the author and category the file held.
                    String author, category;
                    const File target = safe->pm_.getStore().presetFile(
                        safe->pm_.getCurrentBank(), clean);
                    if (const auto xml = MarsDSP::Presets::PresetStore::loadPresetFile(target))
                    {
                        author = xml->getStringAttribute(MarsDSP::Presets::kPresetAuthorProp);
                        category = xml->getStringAttribute(MarsDSP::Presets::kPresetCategoryProp);
                    }
                    target.deleteFile();
                    if (! safe->pm_.saveAs(clean, author, category))
                        NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::WarningIcon,
                            "Save Failed", "The preset could not save.");
                    safe->refreshName_();
                }
            }));
    }
    else
    {
        if (! pm_.saveAs(clean, "", ""))
            NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::WarningIcon,
                "Save Failed", "The preset could not save.");
        refreshName_();
    }
}

void PresetBar::doRename_()
{
    auto* w = new AlertWindow("Rename Preset", "Enter a new name for the preset.", MessageBoxIconType::NoIcon);
    w->addTextEditor("name", pm_.getCurrentName(), "Name:");
    w->addButton("Rename", 1, KeyPress(KeyPress::returnKey));
    w->addButton("Cancel", 0, KeyPress(KeyPress::escapeKey));

    const auto safe = SafePointer<PresetBar>(this);
    w->enterModalState(true, ModalCallbackFunction::create([safe, w](int r) {
        if (safe != nullptr && r == 1)
        {
            const String name = w->getTextEditorContents("name");
            safe->completeRename_(name);
        }
    }), true);
}

void PresetBar::completeRename_(const String& name)
{
    const String clean = MarsDSP::Presets::PresetStore::sanitiseName(name);
    if (clean.isEmpty())
    {
        NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::WarningIcon,
            "Invalid Name", "The preset name is empty. Enter a valid name.");
        return;
    }

    if (! pm_.renameCurrent(clean))
        NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::WarningIcon,
            "Rename Failed", "A preset with that name already exists.");
    refreshName_();
}

void PresetBar::doDelete_()
{
    const auto safe = SafePointer<PresetBar>(this);
    NativeMessageBox::showYesNoBox(MessageBoxIconType::WarningIcon,
        "Delete Preset", "Delete the preset \"" + pm_.getCurrentName() + "\"?",
        nullptr,
        ModalCallbackFunction::create([safe](int r) {
            if (safe != nullptr && r == 1)
                safe->completeDelete_();
        }));
}

void PresetBar::completeDelete_()
{
    if (! pm_.deleteCurrent())
        NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::WarningIcon,
            "Delete Failed", "The preset could not delete.");
    refreshName_();
}

void PresetBar::doExport_()
{
    exportChooser_ = std::make_unique<FileChooser>("Export Preset",
        File::getSpecialLocation(File::userDocumentsDirectory), "*.chronos");
    const auto safe = SafePointer<PresetBar>(this);
    exportChooser_->launchAsync(FileBrowserComponent::saveMode
        | FileBrowserComponent::canSelectFiles
        | FileBrowserComponent::warnAboutOverwriting, [safe](const FileChooser& fc) {
        if (safe == nullptr) return;
        File file = fc.getResult();
        if (file == File()) return;

        // Append the preset extension when the name has none, so the
        // exported file is visible to the import filter.
        if (file.getFileExtension().isEmpty())
            file = file.getSiblingFile(file.getFileName()
                + String(MarsDSP::Presets::kPresetExtension));

        const String xml = safe->pm_.copyPresetXml();
        if (! file.replaceWithText(xml))
            NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::WarningIcon,
                "Export Failed", "The preset could not export to that location.");
    });
}

void PresetBar::doImport_()
{
    importChooser_ = std::make_unique<FileChooser>("Import Preset",
        File::getSpecialLocation(File::userDocumentsDirectory), "*.chronos");
    const auto safe = SafePointer<PresetBar>(this);
    importChooser_->launchAsync(FileBrowserComponent::openMode | FileBrowserComponent::canSelectFiles, [safe](const FileChooser& fc) {
        if (safe == nullptr) return;
        const File file = fc.getResult();
        if (file == File()) return;

        safe->confirmDiscardChanges_([safe, file]
        {
            if (safe == nullptr) return;
            if (! safe->pm_.loadPreset(file))
                NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::WarningIcon,
                    "Import Failed",
                    "The preset file could not load. " + safe->pm_.getLastError());
            safe->refreshName_();
        });
    });
}

} // namespace MarsDSP::GUI
