# InnerSpeech_EEGFMRI
This repo contains all the code for preprocessing the raw EEG and FMRI files for usage

## Preprocessing Steps for FMRI using SPM GUI
#### SPM Version 12 was used to generate the included .mat files
1.	Calculate VDM 
Inputs: Phase Image, Magnitude Image, Anatomical Image, EPI for Unwrap
Outputs: Voxel Displacement Map
Parameters: Echo times [4.92 7.38], Total EPI readout time 
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


General Information:
1.  In anat folder anatomical images are present T_subXX.nii
2. Phase images end with XXXXX_ph.nii and is available in GRE7 for Session 1 and GRE6 for session 2
3. Magnitude images are found in GRE6 and GRE5 for both sessions respectively. There are 2 magnitude files (XXXX_E1.nii and XXXX_e2.nii) in each folder. We use XXXX_e1.nii
4. The functional images (EPI for Unwrap) is available in CMRR folder across al sessions in all subjects.
They vary from 1039 to 1100 so extra attention must be paid for mapping

## Preprocessing Steps for EEG
Vibha and Raj put the text for preprocessing here!! as soon as possible 
