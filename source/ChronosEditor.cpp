#include "ChronosProcessor.h"
#include "ChronosEditor.h"

//==============================================================================
ChronosEditor::ChronosEditor (ChronosProcessor& p) : AudioProcessorEditor (&p), pref (p)
{
    ignoreUnused (pref);
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    setSize (400, 300);
}

ChronosEditor::~ChronosEditor()
{
}

//==============================================================================
void ChronosEditor::paint (Graphics& g)
{
}

void ChronosEditor::resized()
{
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..
}
