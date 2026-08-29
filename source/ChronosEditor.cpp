#include "ChronosEditor.h"

ChronosEditor::ChronosEditor(ChronosProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p)
{
    setLookAndFeel(&lnf_);

    setResizable(true, true);
    setResizeLimits(600, 360, 1500, 900);
    getConstrainer()->setFixedAspectRatio(1000.0 / 600.0);
    setSize(1000, 600);
}

ChronosEditor::~ChronosEditor()
{
    setLookAndFeel(nullptr);
}

void ChronosEditor::paint(Graphics& g)
{
    g.fillAll(MarsDSP::GUI::Colours::background);
}

void ChronosEditor::resized()
{
}
