import numpy as np
import pickle 
import mne
from nilearn import datasets, surface
subjects_dir = mne.datasets.sample.data_path() / 'subjects'

fs_version = 'fsaverage'
datasets.fetch_surf_fsaverage(fs_version)
import nibabel as nib
subjects_dir = mne.datasets.sample.data_path() / 'subjects'
labels = mne.read_labels_from_annot(
    fs_version, 'HCPMMP1', 'both', subjects_dir=subjects_dir)

## Define the names of the regions in the atals

indices_dict = pickle.load(open('/DS/brn-vision/work/smr_proj/encoding-model-scaling-laws/datasets/subject_NCs/late_language_rois.p', 'rb'))

early_visual_rois = ['V1','V2']
early_indices = [indices_dict[r] for r in early_visual_rois]

vwfa_rois = ['PH','TE2P']
vwfa_indices = [ 111, 112]

early_auditory_rois = ['A1','LBelt','PBelt','MBelt','RI','A4']
early_auditory_indices = [indices_dict[r] for r in early_auditory_rois]

primary_auditory_rois = ['A1']
primary_auditory_indices = [indices_dict[r] for r in primary_auditory_rois]

secondary_auditory_rois = ['STV']
secondary_auditory_indices = [indices_dict[r] for r in secondary_auditory_rois]

late_language_rois = ['A5','44','45', 'IFJa', 'IFSp', 'PGi', 'PGp', 'PGs', 'TPOJ1', 'TPOJ2', 'TPOJ3', 'STGa','STSda', 'STSdp','TA2', 'STSva', ]
late_language_indices = [indices_dict[r] for r in late_language_rois]

main_language_rois = ['EV','VWFA','EAC','LL','A1']
angular_gyrus_rois = ['PGi', 'PGp', 'PGs', 'TPOJ2', 'TPOJ3']
ltc_rois = ['TPOJ1', 'STGa','STSda', 'STSdp','A5', 'STSva', ]
ifg_mfg_rois = ['44','45', 'IFJa', 'IFSp']

angular_gyrus_indices = [indices_dict[r] for r in angular_gyrus_rois]
ltc_indices = [indices_dict[r] for r in ltc_rois]
ifg_mfg_indices = [indices_dict[r] for r in ifg_mfg_rois]

main_language_indices = [early_indices, vwfa_indices, early_auditory_indices, late_language_indices, primary_auditory_indices, angular_gyrus_indices, ltc_indices, ifg_mfg_indices]


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


def get_fil_acc(corrs_unnorm, cc_max, cc_thr=0.4, roi_inds=None):
    acc = []
    if roi_inds:
        corrs_unnorm = corrs_unnorm[roi_inds]; cc_max = cc_max[roi_inds]
        
    for ei, i in enumerate(cc_max):
        if i <= cc_thr:
            acc.append(np.nan)
        else:
            acc.append(corrs_unnorm[ei])
    acc = np.array(acc)
    return acc



from nilearn import datasets, surface
fsaverage = datasets.fetch_surf_fsaverage(fs_version)
def get_hm_surf(im):
    surf_data_lh = surface.vol_to_surf(
        im,
        surf_mesh=fsaverage["pial_left"],
        inner_mesh=fsaverage["white_left"],
        interpolation="linear" 
    )

    surf_data_rh = surface.vol_to_surf(
        im,
        surf_mesh=fsaverage["pial_right"],
        inner_mesh=fsaverage["white_right"],
        interpolation="linear" )
    
    return surf_data_lh, surf_data_rh

def get_nib_im(vol,
               subj,
               pdir='../datasets/affine_transforms'): # path for affine matrices used to project to MNI space - already precomputed
    affine = np.load(f'{pdir}/affine_s{subj}.npy')
    nib_im = nib.Nifti1Image(vol.volume.data.squeeze().T, affine)
    return nib_im



def get_norm_lan_scores(surf_data_lh, surf_data_rh, model_name):
    normalized_scores_text_lan = {model_name:[]}

    for eachroi in np.arange(len(main_language_indices)):
        temp2 = []
        for subroi in main_language_indices[eachroi]:
            temp1 = []
            temp = []
            # print(model_name, labels[subroi])
            lh = np.nan_to_num(surf_data_lh[labels[subroi].vertices])
            rh = np.nan_to_num(surf_data_rh[labels[subroi + 180].vertices ])

            lh_rh = np.concatenate([lh, rh],axis=0)
            temp.append(np.mean(lh_rh))

            temp1.append(np.mean(temp))
            temp2.append(np.array(temp1))
            # print(f'{labels[subroi].name} mean: {temp2}')
        normalized_scores_text_lan[model_name].append(np.mean(np.array(temp2), axis=0))
    normalized_scores_text_lan = np.array(list(normalized_scores_text_lan[model_name]))
    return normalized_scores_text_lan.squeeze()


