import os, sys
from typing import List, Optional
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
import torchaudio
import collections
from torch.nn.utils.rnn import pad_sequence
from ridge_utils.interpdata import lanczosinterp2D
import joblib, h5py
import torch
from ridge_utils.dsutils import make_word_ds
import cortex


zscore = lambda v: (v - v.mean(0)) / v.std(0)
zscore.__doc__ = """Z-scores (standardizes) each column of [v]."""
zs = zscore

## Matrix corr -- find correlation between each column of c1 and the corresponding column of c2
mcorr = lambda c1, c2: (zs(c1) * zs(c2)).mean(0)
mcorr.__doc__ = """Matrix correlation. Find the correlation between each column of [c1] and the corresponding column of [c2]."""

### Ignore irrelevant warnings that muck up the notebook
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

## default configs
TARGET_SAMPLE_RATE = 16000
trim_start = 50 # Trim 50 TRs off the start of the story
trim_end = 5 # Trim 5 off the back
ndelays = 4 # We use 4 FIR delays (2 seconds, 4 seconds, 6 seconds, 8 seconds)
delays = range(1, ndelays + 1)
SUBJ_LIST = [1, 2, 3, ]

grids = joblib.load("../datasets/story_data/grids_huge.jbl") # Load TextGrids containing story annotations
trfiles = joblib.load("../datasets/story_data/trfiles_huge.jbl") # Load TRFiles 

# We'll build an encoding model using this set of stories for this tutorial.
train_stories = ['adollshouse', 'adventuresinsayingyes', 'alternateithicatom', 'avatar', 'buck', 'exorcism',
            'eyespy', 'fromboyhoodtofatherhood', 'hangtime', 'haveyoumethimyet', 'howtodraw', 'inamoment',
            'itsabox', 'legacy', 'naked', 'odetostepfather', 'sloth',
            'souls', 'stagefright', 'swimmingwithastronauts', 'thatthingonmyarm', 'theclosetthatateeverything',
            'tildeath', 'undertheinfluence']

test_stories = ["wheretheressmoke"]

# Make datasequence for story
wordseqs = make_word_ds(grids, trfiles)

story_lists = np.load('../datasets/story_lists.npy') #list(wordseqs.keys())


def get_voxels_nc_mask(subject, thr=0.4):
    sub_nc = np.load(f'../subject_NCs/UTS0{subject}.npy')
    sub_neur_mask = np.where(sub_nc > thr)[0]
    out_dim = len(sub_neur_mask)  # Desired output dimension
    return sub_neur_mask, out_dim

def load_story_data(story_name,
                    subject,
                    fmri_dir='../../ds003020/derivative/preprocessed_data/',
                    story_dir='../datasets/processed_stim_data_dp',):
    wav_tensor = torch.tensor(np.load(os.path.join(story_dir, f"{story_name}/wav.npy")))
    fmri_tensor = torch.tensor(_load_h5py(os.path.join(fmri_dir,f'UTS0{subject}', f"{story_name}.hf5"))).float()
    return wav_tensor, fmri_tensor
    

def load_fmri_story(story_name, subject, key=None,
                    fmri_dir='../../ds003020/derivative/preprocessed_data/',
                    ):   

    data = dict()
    with h5py.File(os.path.join(fmri_dir,f'UTS0{subject}', f"{story_name}.hf5")) as hf:
        if key is None:
            for k in hf.keys():
                print("{} will be loaded".format(k))
                data[k] = np.array(hf[k])
        else:
            data[key] = hf[key]
    return data# mask out the nans

def _load_h5py(file_path, key=None):   

    data = dict()
    with h5py.File(file_path) as hf:
        if key is None:
            for k in hf.keys():
                print("{} will be loaded".format(k))
                data[k] = list(hf[k])
        else:
            data[key] = hf[key]
    return np.nan_to_num(np.array(data['data']))# mask out the nans


def get_roi_fmri_mask(subject, roi=None):
    '''
    for the given ROI, get the indices of the voxels in the subject's brain predictions/fmri data
    '''
    sub_nc = np.load(f'../subject_NCs/UTS0{subject}.npy')
    
    sub_vol = cortex.Volume(sub_nc, f'UTS0{subject}', f'UTS0{subject}_auto', vmin=0, vmax=1, cmap='fire') # another way is to load `mask_thick.nii.gz` from transforms
    if roi is None:
        roi_mask = cortex.utils.get_roi_masks(f'UTS0{subject}', f'UTS0{subject}_auto', )
    else:
        roi_mask = cortex.utils.get_roi_masks(f'UTS0{subject}', f'UTS0{subject}_auto', roi_list= [roi]) # get the mask for the ROI volume
    l = np.where(sub_vol.mask.ravel())[0] # get the indices of the voxels in the subject mask volume (len = len(fmri data))
    
    if roi is not None:
        id_mask = np.where(roi_mask[roi].ravel()>0)[0] # get the indices of the given voxels in the ROI
        
        fmri_roi_msk = np.flatnonzero(np.in1d(l, id_mask)) # now get the indices in the fmri data for that ROI
        return fmri_roi_msk
    else:
        fmask = {}
        for roi in roi_mask:
            id_mask = np.where(roi_mask[roi].ravel()>0)[0]
            fmri_roi_msk = np.flatnonzero(np.in1d(l, id_mask))
            fmask[roi] = fmri_roi_msk
        return fmask # return all the masks for the ROIs

def spe_and_cc_norm(orig_data, data_pred, data_norm=True, max_flooring=None):
    '''
    Computes the signal power explained and the cc_norm of a model given the observed and predicted values
    Assumes normalization unless data_norm is set to False

    According to Schoppe: https://www.frontiersin.org/articles/10.3389/fncom.2016.00010/full
    '''
    y = np.mean(orig_data, axis=0)
    num_trials = len(orig_data)
    if not data_norm:
        variance_across_time = np.var(orig_data, axis=1, ddof=1)
        TP = np.mean(variance_across_time, axis=0)
    else:
        TP = np.zeros(orig_data.shape[2]) + 1
    SP = (1 / (num_trials-1)) * ((num_trials * np.var(y, axis=0, ddof=1)) - TP) 
    SPE_num = (np.var(y, axis=0, ddof=1) - np.var(y - data_pred, axis=0, ddof=1)) 
    SPE = (np.var(y, axis=0, ddof=1) - np.var(y - data_pred, axis=0, ddof=1)) / SP
    y_flip = np.swapaxes(y, axis1=0, axis2=1)
    data_flip = np.swapaxes(data_pred, axis1=0, axis2=1)
    covs = np.zeros(y_flip.shape[0])
    for i, row in enumerate(y_flip):
        covs[i] = np.cov(y_flip[i], data_flip[i])[0][1]
    cc_norm =  np.sqrt(1/SP) * (covs / np.sqrt(np.var(data_pred, axis=0, ddof=1)))
    cc_max = None
    if max_flooring is not None:
        cc_max = np.nan_to_num(1 / (np.sqrt(1 + ((1/num_trials) * ((TP/SP)-1)))))
        #cc_max = np.maximum(cc_max, np.zeros(cc_max.shape) + max_flooring)
        corrs = np.zeros(y_flip.shape[0])
        for i, row in enumerate(y_flip):
            corrs[i] = np.corrcoef(y_flip[i], data_flip[i])[0][1]
        cc_norm = corrs / cc_max
    return SPE, cc_norm, cc_max, corrs

