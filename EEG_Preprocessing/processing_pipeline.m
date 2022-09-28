% EEGLAB history file generated on the 29-Jun-2022
% ------------------------------------------------

%%%1. Importing the Biosemi raw data file. Using reference channel 48 (Cz)
%%%upon import

EEG.etc.eeglabvers = '2022.0'; % this tracks which version of EEGLAB is being used, you may ignore it
EEG = pop_readbdf('subject02_session01.bdf', [1 1306] ,73,48);
EEG = eeg_checkset( EEG );

%%2. Remove *real* channel mean:

for numChans=1:size(EEG.data,1)
    EEG.data(numChans,:)=single(double(EEG.data(numChans,:))-mean(double(EEG.data(numChans,:))));
end

%%3. Filter the data
EEG = pop_eegfiltnew(EEG, 'locutoff',0.1,'hicutoff',50,'plotfreqz',1);
EEG = eeg_checkset( EEG );

%%4. Put channel location 
EEG=pop_chanedit(EEG, 'lookup','location of file');
EEG = eeg_checkset( EEG );


%%Re-referencing. check 'compute aaverage refrence'. Exclude channel
%%indices[65-72]. Add current refrence channel back to the data
EEG = pop_reref( EEG, [],'exclude',[65:72] );
EEG = eeg_checkset( EEG );

%%decompose by ICA and remove components (manually) which correspond to EOG (runica)
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on');
EEG = eeg_checkset( EEG );

EEG = pop_iclabel(EEG, 'default');
EEG = eeg_checkset( EEG );

EEG = pop_subcomp( EEG, [components to delete], 0);
EEG = eeg_checkset( EEG );

%%Extracting epoch using handwritten scripts

Program_for_extracting_label_from_gT.m

