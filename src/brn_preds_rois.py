from nilearn import datasets, surface
import torch
import pickle
import numpy as np
import cortex
fsaverage = datasets.fetch_surf_fsaverage('fsaverage6')
from brn_align_utils import *


## NOTE: for handling some torch errors in my code. PLEASE comment out if not needed.
import sys, types, importlib, numpy as np
def _alias(old, new):
    try:
        mod = importlib.import_module(new)
    except ModuleNotFoundError:
        return
    parent, _, _ = old.rpartition('.')
    if parent and parent not in sys.modules:
        sys.modules[parent] = types.ModuleType(parent)
    sys.modules[old] = mod

_alias('numpy._core.numeric', 'numpy.core.numeric')

## load subject NCs for normalization
NCs = {}
for x in [1, 2, 3, 4, 5, 6, 7, 8]:
    subj_data = np.load((f'../datasets/subject_NCs/UTS0{x}.npy'), allow_pickle=True)
    NCs[x] = subj_data

def read_brain_preds(subj, pred_key):
    # try:
    base_prds = torch.load(f'{pred_key}/subject_{subj}/results.pt', map_location='cpu')
    return base_prds['corr']

    # except Exception as e:
    #     print(f'Error {e} in subj {subj}')
    #     return

def get_vol_base_(subj, 
                  pred_key,
                  pred_read_func, # the custom function to read predictions from disk - look above for an example
                  cc_thr=0.05,
                  ):

    base_prds = pred_read_func(subj, pred_key)
    base_arr = get_fil_acc(base_prds, NCs[subj], cc_thr)

    diff_arr = base_arr
        
    vol = cortex.Volume(np.nan_to_num(diff_arr/NCs[subj]), f'UTS0{subj}', f'UTS0{subj}_auto')
    nib_im = get_nib_im(vol, subj)
    
    return nib_im


def get_vol_hm_(subj, 
                pred_key,
                pred_read_func
):
    im = get_vol_base_(subj,
                        pred_key=pred_key,
                        pred_read_func=pred_read_func)
    surf_data_lh, surf_data_rh = get_hm_surf(im)
    return surf_data_lh, surf_data_rh

def get_norm_scores_base(subj, 
                        pred_key,
                        model_name,
                        pred_read_func
):
    surf_data_lh_b, surf_data_rh_b = get_vol_hm_(subj,
                                                 pred_key=pred_key,
                                                 pred_read_func=pred_read_func,
                                                 )
    norm_scores_base = get_norm_lan_scores(surf_data_lh_b, surf_data_rh_b, model_name=model_name)
    
    return norm_scores_base


def get_subjs_layers_base(subj_list=[3], # subject list 
                          pred_key='../qa_feats/original/', # path to brain predictions
                          model_name='qa', # model name for storing scores, not really used in this case
                          pred_read_func=read_brain_preds
):
    region_keys = ['early_visual', 'VWFA', 'early_auditory', 'late_language', 'primary_auditory', 'angular_gyrus', 'ltc', 'ifg_mfg']
    base_scores = {k:{s:{} for s in subj_list}  for k in region_keys }    
    for subj in subj_list:
        try:
            norm_scores_base = get_norm_scores_base(subj=subj,
                                                    pred_key=pred_key,
                                                    model_name=model_name,
                                                    pred_read_func=pred_read_func)
            for i, k in enumerate(region_keys):
                base_scores[k][subj] = (norm_scores_base[i])  
        except Exception as e:
            print(f'Error {e} in subj {subj}')
            continue
        
    return base_scores




if __name__ == "__main__":
    ## example usage on previously computed brain predictions
    parent_dir = '/BRAIN/ckolling-phd/work/git/semantic_llm2brain/logs/moth_radio/brain_encode/'
    postag_type = 'noun_verb_adv_adj_num_pron_propn'
    experiments_path = {
        "qa35": "qa_feats/subset_qa/",
        "up-to-tag": f"binder_feats/prompt_v1/scale_0-6/{postag_type}/up_to_tag/llama3.1-8b-instruct/subset_qa",
    }
    rois_compare_results = {} 
    for mname in experiments_path:
        pred_key = f'{parent_dir}{experiments_path[mname]}'
        d = get_subjs_layers_base(subj_list=[1, 2], pred_key=pred_key, pred_read_func=read_brain_preds)
        rois_compare_results[mname] = d
    
    print(rois_compare_results['up-to-tag']) # the dict is formatted as {region_name: {subject_id1: scores_array, subject_id2: scores_array, ...}, ...}




    