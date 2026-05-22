.. _loading-data:

Loading previously recorded data
================================

In case you previously recorded song data with EvTAF (Tumer & Brainard, 2007), or with any other program producing either
``.wav`` or ``.cbin`` files, they can be loaded into the MooveGUI as well. The only requirement is a respective ``.not.mat`` 
file for each song file. A ``.rec`` file is optional, but if available, it will be used to show feedback and catch trials. 
Missing ``.rec`` files will be created once you open the file in MooveGUI, feedback and catch trial
information will per default be set to zero.

Furthermore, in case your already existing ``.rec`` files are missing the two info lines for hand segmentation 
and classification, the respective lines with **Hand Segmented** and **Hand Classified** 
(see section *REC file and Feedback Information*) will be added when opening the file in MooveGUI. 
This will not change any of the existing information in the file, but simply add the two lines at the correct positions. 

Eventually, if not done so before, move the data folder into your *rec_data* folder in ``.moove`` 
(pay attention to the correct folder structure, see section *MooveTAF*) and start the MooveGUI. 
Note that in order to show feedback and catch trials correctly, the info lines in the ``.rec`` files 
must be formatted identically to the ones created by MooveTaf (see section *REC file and Feedback Information*).

As your old files will presumably contain different dB-values, you might want to adjust the slider values to correctly 
display the spectrogram (see section *MooveGUI*).

