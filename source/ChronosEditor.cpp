#include "ChronosProcessor.h"
#include "ChronosEditor.h"

//==============================================================================
ChronosEditor::ChronosEditor (ChronosProcessor& p) : AudioProcessorEditor (&p), pref (p)
{
    ignoreUnused (pref);
    setSize(1000, 600);
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
