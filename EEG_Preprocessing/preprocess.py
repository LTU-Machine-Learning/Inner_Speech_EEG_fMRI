import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import pdb
import json
import argparse
import ast

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

import mne
from mne import Epochs, pick_types, find_events, pick_types, set_eeg_reference
from mne.io import concatenate_raws, read_raw_bdf
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs, corrmap

# from autoreject import AutoReject
# from autoreject import get_rejection_threshold
# from autoreject import Ransac

# ar = AutoReject()
# rsc = Ransac()

from tqdm import tqdm
from pprint import pprint
from collections import defaultdict

from pdb import set_trace as db

# all externals apparently used (EXG7 and EXG8 recorded junk)
eog = ['EXG1','EXG2','EXG3','EXG4','EXG5','EXG6', 'EXG7', 'EXG8']

class Preprocess:
    
    def __init__(self, _, paths,
        #model_name = None,
        #channels_transform = False,
        #frequency_bands_transform = False,
        ):

        self.args = _
        #self.model_name = model_name
        #self.channels_transform = channels_transform # either all channels, or selected channels around Broca's Wernicke's area
        #self.frequency_bands_transform = frequency_bands_transform # either all frequency bands, or high gamma only
        # self.window_transforms = {
        # 'CNN':lambda X,y : (X, y), # D x H x W = sensors x freqbands x samples
   
    def clean_data(self):

        bdf_labels = [self.args['id'] if self.args['id'] else [1,2,3,5]]

        for i in bdf_labels:

            ### Read data ### 

            raw_fname = paths['data'].format(self.args['id'])
            raw = read_raw_bdf(
                raw_fname,
                preload=True,
                eog=eog,
                stim_channel='Status',
                verbose=self.args['verbose'])

            # get some basic visual info about the data:

            print('Data type: {}\n\n{}\n'.format(type(raw), raw))
            print('Sample rate:', raw.info['sfreq'], 'Hz')
            print('Size of the matrix: {}\n'.format(raw.get_data().shape))
            print(raw.info)
            print('\n\n {}\n'.format(raw.get_data()))

            # assign indices:

            eeg_channel_indices = mne.pick_types(raw.info, eeg=True)
            eog_channel_indices = mne.pick_types(raw.info, eog=True)
            stim_channel_indices = mne.pick_types(raw.info, stim=True)

            # get stimuli timings:

            timings = paths['timings'].format(self.args['id'])
            timings = pd.read_csv(timings, encoding='utf16', skiprows=1, sep='\t')
            #timings = timings.loc[:, timings.columns.values[32:]]
            
            ### Events ###

            fixation_onset = timings['fixation.OnsetTime']
            stimulus_onset = timings['stimulus.OnsetTime']
            rest_onset = timings['rest.OnsetTime']
            trigger_id = timings['imgTrigger']
            displayed_word = timings['InnerWord']
            semantic_class = timings['Condition']
            
            event_dict = {i:j for i,j in set(zip(trigger_id,displayed_word))}
            event_dict[1] = 'fixation'
            event_dict[2] = 'rest'
            event_id = {i:j for j,i in event_dict.items()} # needed for epoching later
            events = []
            # Fix instances where there are anomolous preceding trigger events: 
            for i in range(len(stimulus_onset)):
                events.append(np.array([fixation_onset[i],0,1]))
                events.append(np.array([stimulus_onset[i],0,trigger_id[i]]))
                events.append(np.array([rest_onset[i],0,2]))
            events = np.array(events)
            _events = mne.find_events(raw, stim_channel='Status')
            events = _events[np.where(_events[:,2]==events[1][2])[0][0]-1:]

            ### Annotations ###

            annot_from_events = mne.annotations_from_events(
                events=events, event_desc=event_dict, sfreq=raw.info['sfreq'])
            
            raw.set_annotations(annot_from_events) # assign annotations

            ### Basic info processing ###

            raw.drop_channels(['EXG7','EXG8']) # Unused; junk channels.

            # Sanity-check via STIM channel:
            # stim = raw.copy().pick(['Status']).load_data() # copy, otherwise modifies in place
            # data, times = stim[:]

            biosemi_montage = mne.channels.make_standard_montage('biosemi64')
            raw.set_montage(biosemi_montage)

            # Set mastoid references:
            raw.set_eeg_reference(
                ref_channels = ['EXG1', 'EXG2'])

            ### Low-frequency drifts ###

            # Sanity check that our data does NOT have DC drift applied online (during data collection)
            
            if self.args['plot']:
                raw.plot(
                duration = 60,
                remove_dc = False) # CONFIRMED - DC drift was not automatically applied

            ### Power line noise ###

            # Sanity check that our data actually does have line noise artifacts:

            if self.args['plot']:
                raw.plot_psd(
                area_mode='range',
                tmax = np.inf,
                picks = 'eeg',
                average = False) # CONFIRMED - notch filter was not automatically applied

            # notch filter the data at 50, 100 and 150 to remove the 50 Hz line noise and its harmonics:
            
            notches = np.arange(50, 100, 150)
            raw.notch_filter(
                notches,
                picks = 'eeg',
                filter_length = 'auto',
                phase = 'zero-double', # 'zero'
                fir_design = 'firwin')

            # Filter the data to remove low-frequency drifts (i.e. DC drifts):

            filt_raw = raw.copy().filter(l_freq=1., h_freq=None)

            # ICA is not deterministic (e.g., the components may get a sign flip on different runs,
            # or may not always be returned in the same order), so we should set a seed.

            ica = mne.preprocessing.ICA(
                n_components=15, # for now, but should = number of channels
                noise_cov=None,
                random_state=123, # seed
                method='fastica',
                fit_params=None,
                max_iter='auto',
                allow_ref_meg=False,
                verbose=self.args['verbose'],
                )

            ica.fit(filt_raw, # fit to filt_raw, apply to raw
                picks='data',
                start=None,
                stop=None,
                decim=None, # every Nth sample
                reject=None, # peak-to-peak amplitudes
                flat=None,
                tstep=2.0, 
                reject_by_annotation=True,
                verbose=self.args['verbose'])

            if self.args['plot']:
                ica.plot_sources(raw) # note applied to original raw
                ica.plot_components(inst=raw)
                ica.plot_scores(eog_scores) # barplot of ICA component "EOG match" scores
                ica.plot_properties(raw, picks=eog_indices) # plot diagnostics
                # if we have evoked; good to sanity-check:
                # eog_evoked = create_eog_epochs(filt_raw).average()
                # eog_evoked.apply_baseline(baseline=(None, -0.2))
                # ica.plot_sources(eog_evoked) if we have evoked; good to sanity-check
                
            # MANUALLY EXCLUDE BASED ON VISUAL INSPECTION:
            #ica.exclude = [1] # IC 1 is EOG artifacts, for example
            #ica.apply(raw)
            
            # AUTOMATICALLY EXCLUDE BASED ON EOG CHANNELS:
            ica.exclude = []
            # find which ICs match the EOG pattern
            eog_indices, eog_scores = ica.find_bads_eog(filt_raw, verbose=self.args['verbose'])
            ica.exclude = eog_indices  
            ica.apply(raw)

            ### Lazy defaults ###

            tmin = -.2
            tmax = .5

            # make metadata for use with Epochs (mimics epoching process)
            # n.b If attaching the generated metadata to Epochs and intending
            # to include only events that fall inside those epochs' time interval,
            # pass the same tmin and tmax values here as used for the epochs.
            
            metadata, events, event_id = mne.epochs.make_metadata(
                events = events,
                event_id = event_id,
                tmin = tmin,
                tmax = tmax,
                sfreq = raw.info['sfreq'],
                row_events=None,
                keep_first=None,
                keep_last=None,
                )

            # mne.event.shift_time_events - will need this for experiments

            ### Filtering the raw data with ICs zero-ed out ###
            
            # Remove the hf/lf components, if needed. AT LEAST need to do an lf for DC drift.
            
            raw.filter(
                l_freq = 0.5,
                h_freq = 40.,
                method = 'fir',
                fir_window = 'hamming',
                fir_design = 'firwin', # 'firwin2'
                verbose= self.args['verbose'],
                # l_trans_bandwidth=0.5,
                # h_trans_bandwidth=0.5,
                # filter_length='10s',
                # phase='zero-double',
                )
            
            #print('\n\n {}\n'.format(raw.get_data()))
            
            ### Epochs ###

            # When accessing data, epochs are linearly detrended (changable - see below),
            # baseline-corrected and decimated, then projectors are (optionally) applied.

            # N.b. If wanting baseline interval as only one sample, we must use
            # `baseline=(0, 0)` as opposed to `baseline=(None, 0)`
            
            # ALSO Removing EOG artifacts with ICA likely introduces DC offsets,
            # so it's imperative to do baseline correction AFTER ICA and filtering.

            # mne.event.shift_time_events --- will need this for experiments.

            reject = dict(
                eeg=40e-6, # unit: V (EEG channels)
                #eog=250e-6 # unit: V (EOG channels) (we remove these, so ignore)
                )

            epochs = mne.Epochs(
                raw,
                events,
                event_id = event_id,
                tmin = tmin,
                tmax = tmax,
                baseline = (None,0),
                picks = eeg_channel_indices,
                preload = False,
                #reject = reject, # leave for now
                flat = None,
                proj = False,
                decim = 1,
                reject_tmin = None,
                reject_tmax = None,
                detrend = 1, # 0 = constant (DC) detrend, 1 = linear detrend
                on_missing = 'raise',
                reject_by_annotation = True,
                metadata = metadata,
                event_repeated = 'error', # 'drop' for coarticulation: keep 1st phone only
                verbose = self.args['verbose'])

            # LEFT HERE FOR FUTURE REFERENCE: To label epochs with blinks you can do:
            #     eog_events = mne.preprocessing.find_eog_events(raw)  
            #     n_blinks = len(eog_events)  
            #     onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25  
            #     duration = np.repeat(0.5, n_blinks)  
            #     description = ['bad blink'] * n_blinks  
            #     annotations = mne.Annotations(onset, duration, description)  
            #     raw.set_annotations(annotations)             

            # epochs.save('epochs.fif',
                 # split_size='2GB',
                 # fmt='double',
                 # overwrite=False,
                 # split_naming='bids', #split files given *-01, *-02 extensions
                 # verbose=True)
                 
            db()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-t", "--train", action="store_true", help="train network from EEG data")
    mode.add_argument("-d", "--test", action="store_true", help="use network to decode test EEG data")
    mode.add_argument("-p", "--plot", action="store_true", help="plot visuals from trained network")
    mode.add_argument("-w", "--debug", action="store_true", help="run in debug mode")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="run with verbose processing")
    # ============model architecture choice=============
    # parser.add_argument("-m", "--model",type=str,choices=Models().models,required=True,
    #     help="Model architecture from one of: {}".format(', '.join(Models().models)))
    # ========thinker-(in)dependent model choice========
    parser.add_argument("--id", type=int, choices=[1,2,3,5],
        help="Subject ID number to train thinker-dependent model (default = thinker-independent model)")
    # =======convenience arguments for data paths=======
    parser.add_argument("-f", "--filter", type=str, choices=['single_bandpass', 'multi_bandpass'], default='multi_bandpass')
    parser.add_argument("-s", "--scope", type=str, choices=['local', 'global'], default='global')
    parser.add_argument("-n", "--normalisation", type=str, choices=['min_max', 'mean_variance'], default='mean_variance')
    parser.add_argument("-b", "--build", type=str, default='') # optional descriptor for model build
    # ===========parameters for convolutions============
    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--out_channels", type=int, default=None)
    parser.add_argument("--kernel_size", type=ast.literal_eval, default=None) # Tuple written as a string; e.g. "(2,3)"
    parser.add_argument("--pool_size", type=ast.literal_eval, default=None)
    parser.add_argument("--stride", type=ast.literal_eval, default=None)
    parser.add_argument("--dilation", type=ast.literal_eval, default=None)
    parser.add_argument("--padding", type=ast.literal_eval, default=None)
    # ===============parameters for data===============
    parser.add_argument("--frequency_bands", default='all', choices=['all', 'high_gamma'])
    # =========user-definable (hyper)parameters=========
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--window_shift", type=int, default=246)    
    parser.add_argument("--crop_size", type=int, default=32)
    parser.add_argument("--crop_shift", type=int, default=16)
    parser.add_argument("--dropout", type=int, default=.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning_rate", type=int, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=99)
    # ==================================================
    args = parser.parse_args()
    
    # ============hardcoded data parameters=============
    test_size = 0.2 # of all data, percentage for test set
    train_size = 0.75 # of the training data, the train/val split
    # which gives us train/test/val of 6/2/2

    # =============softcoded default paths==============
    cwd = os.getcwd()
    IspeecAI = os.path.join(os.path.expanduser("~"),'IspeecAI')
    IspeecAI_EEG = os.path.join(IspeecAI, 'EEG-rec')
    IspeecAI_fMRI = os.path.join(IspeecAI, 'fMRI-rec')
    IspeecAI_EEGproc = os.path.join(IspeecAI, 'EEG-proc')
    IspeecAI_fMRIproc = os.path.join(IspeecAI, 'EEG-proc')
    _a = os.path.join(IspeecAI_EEG,'InnerSpeech-EEG-0014.es3') # E-Prime experiment file
    _b = os.path.join(IspeecAI_EEG,'InnerSpeech-EEG-0014.wndpos') # E-Studio open windows & positions file
    _c = os.path.join(IspeecAI_EEG,'InnerSpeech-EEG-0014-0{}-1.edat3') # E-DataAid experiment data file
    _d = os.path.join(IspeecAI_EEG,'InnerSpeech-EEG-0014-0{}-1.txt') # E-Studio newline-separated timings file
    _e = os.path.join(IspeecAI_EEG,'InnerSpeech-EEG-0014-0{}-1-ExperimentAdvisorReport.xml') # E-Prime advisor file
    timings = os.path.join(IspeecAI_EEG,'InnerSpeech-EEG-0014-0{}-1-export.txt') # tab-separated timings file
    data = os.path.join(IspeecAI_EEG,'subject0{}_session01.bdf') # BioSemi signal data file

    os.chdir(IspeecAI) # jump to correct directory
    print(f"Jumping to working directory: {os.getcwd()}")
    paths = {
        'IspeecAI':IspeecAI,
        'IspeecAI_EEG':IspeecAI_EEG,
        'IspeecAI_fMRI':IspeecAI_EEGproc,
        'IspeecAI_EEGproc':IspeecAI_fMRI,
        'IspeecAI_fMRIproc':IspeecAI_fMRIproc,       
        '_a': _a,
        '_b': _b,
        '_c': _c,
        '_d': _d,
        '_e': _e,
        'timings': timings,
        'data': data,
        }

    # ================preprocess the data=================
    preprocess = Preprocess(vars(args),paths)
    preprocess.clean_data()

    input_dir = os.path.join(IspeecAI_EEGproc,'X')
    target_dir = os.path.join(IspeecAI_EEGproc,'y')
    
    if not os.path.exists('models'):
        os.makedirs('models') 
    if not os.path.exists('results'):
        os.makedirs('results')
    
    models_dir, results_dir = 'models', 'results'
    args.build = args.model + args.build # add descriptor to model .pth save
    
    # =================network setup====================
    # Set if you want to use GPU
    cuda = True if torch.cuda.is_available() else False

    # Initialize model
    models = Models(**vars(args))
    input_size, model = models.get_model()
    network = Network()
    
    if cuda:
            model.cuda()

    # Define loss function:
    loss_function = nn.MSELoss()
    
    # Initialize optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=5e-4) # these are good values for the deep models
    restart_period = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, restart_period, T_mult=1, eta_min=0, last_epoch=-1)

    # Initialize early stopping params
    early_stopping = EarlyStopping(patience=args.patience)

    # ===================data setup=====================
    transform = Transform(
        model_name = args.model,
        frequency_bands_transform = len(args.frequency_bands) < len(all_frequency_bands),
        )
    train_data, val_data, test_data = get_data(input_dir, target_dir) # Get data from user-specified path
    train_sampler, val_sampler = RandomSampler(train_data), SequentialSampler(val_data) # Samplers for training and validation batches
    # ==================================================