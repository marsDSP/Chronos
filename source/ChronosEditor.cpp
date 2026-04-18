#include "ChronosProcessor.h"
#include "ChronosEditor.h"

//==============================================================================
ChronosEditor::ChronosEditor (ChronosProcessor& p)
    : AudioProcessorEditor (&p), pref (p)
{
    ignoreUnused (pref);
    setSize (400, 300);
}

ChronosEditor::~ChronosEditor()
{
}

//==============================================================================
void ChronosEditor::paint (Graphics& g)
{
    ignoreUnused (g);
}

void ChronosEditor::resized()
{
}
