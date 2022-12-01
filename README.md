# Bimodal electroencephalography-functional magnetic resonance imaging dataset for inner-speech recognition
This repository contains all the code for preprocessing the raw EEG and fMRI data of the bimodal Inner Speech Dataset.
The dataset is publicly available in OpenNeuro: https://openneuro.org/datasets/ds004197
```bibtex
Simistira Liwicki, F. et al. "Bimodal dataset on Inner speech". OpenNeuro https://doi:10.18112/openneuro.ds004197.v1.0.2 (2022)
```

The publication is available as a preprint at biorxiv:
https://www.biorxiv.org/content/10.1101/2022.05.24.492109v3 


## Stimulation Protocol
The EEG and fMRI modalities use the same experimental protocol developed with ePrime.

## Preprocessing Steps for fMRI
SPM12 was used to generate the included .mat files

File = `preprocessing.mat`
1.	Calculate VDM 
     - Inputs: Phase Image, Magnitude Image, Anatomical Image, EPI for Unwrap
     - Outputs: Voxel Displacement Map
     - Parameters: Echo times [4.92 7.38], Total EPI readout time 
Echo spacing * number of echos = 33.92
No brain masking
2.	Realign and Unwrap
     - Inputs: All images of the session, Voxel Displacement map
     - Output: uCMRR
3.	Slice Timing
     - Input: Output of previous stage
     - Parameters: TR = 2.16, Number of slices = 68
      - Output: auCMRR
4.	Coregister: Estimate Only
     - Inputs: Reference Image – Mean unwarped image, Source Image- Anatomical Image
     - Output: Coregistered Anatomical Image (rT_)
     
File = `segmentNormalise.mat`

5.  Segmentation:
     - Input: realigned anatomical image (rT_)
     - Parameters: Forward Deformation
     - Output: y_ prefix
6.  Normalisation to MNI Space
     - Input: Deformation Field: y_ file from segmentation step, Images to Write: auCMRR
     - Output: wauCMRR
7.	Smoothing
     - Input: wauCMRR
     - Parameters: Gaussian Kernel [8 8 8]
     - Output: swauCMRR
     
Run `firstlevel.mat` for each subject and session separately to produce the beta images and contrasts. Input to this should also include the rp_ movement parameter files. Researchers may choose to use the beta images produced from this first level step for decoding, or the swauCMRR or wauCMRR for decoding of inner speech.
Run `secondlevel.mat` to compute the group level statistics.


To assess the framewise displacement for technical validation, use the `FWD_script.py`. Inputs for this should include the rp_ movement parameter files. The output is a plot that can be saved to file.


General Information:
1.  In anat folder anatomical images are present `T_subXX.nii`
2. Phase images end with `XXXXX_ph.nii` and is available in GRE7 for Session 1 and GRE6 for session 2
3. Magnitude images are found in GRE6 and GRE5 for both sessions respectively. There are 2 magnitude files (`XXXX_E1.nii` and `XXXX_e2.nii`) in each folder. We use `XXXX_e1.nii`
4. The functional images (EPI for Unwrap) is available in CMRR folder across al sessions in all subjects.
They vary from 1039 to 1100 so extra attention must be paid for mapping

## Preprocessing Steps for EEG

EEGLAB is an open source signal processing environment for electrophysiological signals running on Matlab and Octave. Download it from the official site https://sccn.ucsd.edu/eeglab/download.php

1. Start `Matlab` and use it for navigating to the folder containing `EEGLAB`.

2. Type `eeglab` at the Matlab command prompt and press enter. You can see one pop out window which shows all the functions and modules available here in EEG.

3. Now, we start preprocessing the data. The full description and documentation of the pipeline stepwise has been written in `preprocessing_pipeline.m`

4. The `preprocessing_pipeline.m` will import the data (here it is in .bdf format) and do the preprocessing steps like filtering of data, referencing, removal of noise by ICA decomposition etc.

5. During the ICA decomposition part, we delete the components which correspond to motor-related artifacts such as blinking, jaw, neck, arm, or upper back. We performed this step by manual observation the scores shown by the ICA decomposition tool. 

6. After cleaning the data, we segment/epoch the continuous data using the script `Program_for_extracting_label_from_gT.m`
     - Channels: 64
     - Sampling rate: 512Hz
     - Number of stimuli: 960 which includes classes (320), fixation (320), relaxation (320)

6. `Program_for_extracting_label_from_gT.m` will make the data suitable for further tasks; plotting and classification.

**Note: documentation is included in the scripts.

## Citing this work
```bibtex
@article {Simistira Liwicki2022.05.24.492109,
	author = {Simistira Liwicki, Foteini and Gupta, Vibha and Saini, Rajkumar and De, Kanjar and Abid, Nosheen and Rakesh, Sumit and Wellington, Scott and Wilson, Holly and Liwicki, Marcus and Eriksson, Johan},
	title = {Bimodal electroencephalography-functional magnetic resonance imaging dataset for inner-speech recognition},
	elocation-id = {2022.05.24.492109},
	year = {2022},
	doi = {10.1101/2022.05.24.492109},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/11/30/2022.05.24.492109},
	eprint = {https://www.biorxiv.org/content/early/2022/11/30/2022.05.24.492109.full.pdf},
	journal = {bioRxiv}
}

```
