Dialog.create("Slice stack");
Dialog.addString("From:", 1);
Dialog.addString("To:", nSlices);
Dialog.show();

from = Dialog.getString();
to = Dialog.getString();

run("Make Substack...", "  slices="+from+"-"+to);