import numpy as np  
import sys
import os
from residuals_text_speech import * 

import torch 
import argparse 
import matplotlib.pyplot as plt 
from extract_speech_features import extract_speech_features
from wav2vec_linear import Wav2VecLinear
from eval_utils import *
from ridge_utils.interpdata import lanczosinterp2D
import ridge_utils.npp
from ridge_utils.ridge import bootstrap_ridge
from ridge_utils.util import make_delayed
import pickle
from glob import glob
import sys 


#These files contains low-level textual and speech features
def load_low_level_textual_features():
    # 'letters', 'numletters', 'numphonemes', 'numwords', 'phonemes', 'word_length_std'
    base_features_train = np.array(h5py.File('../features_trn_NEW.hdf','r+'))
    base_features_val = np.array(h5py.File('../features_val_NEW.hdf','r+'))
    return base_features_train, base_features_val

def load_low_level_speech_features(lowlevelfeature):
    # 'diphone', 'powspec', 'triphone'
    if lowlevelfeature in ['diphone', 'powspec', 'triphone']:
        df = h5py.File('../speech-llm-brain/Low-level-features/features_matrix.hdf')
        base_features_train = np.array(df[lowlevelfeature+'_train'])
        base_features_val = np.array(df[lowlevelfeature+'_test'])
    elif lowlevelfeature in 'articulation':
        base_features_train = np.load('../speech-llm-brain/Low-level-features/articulation_train.npy')
        base_features_val = np.load('../speech-llm-brain/Low-level-features/articulation_test.npy')
    return base_features_train, base_features_val



def get_residuals(stimulus_features_train, stimulus_features_test, lowlevelfeature):
    if lowlevelfeature in ['letters', 'numletters', 'numphonemes', 'numwords', 'phonemes', 'word_length_std']:
        base_features_train, base_features_val = load_low_level_textual_features()
        residual_features = residuals_textual(base_features_train, base_features_val, stimulus_features_train, stimulus_features_test)
    elif lowlevelfeature in ['powspec', 'diphone', 'triphone','articulation']:
        base_features_train, base_features_val = load_low_level_speech_features(lowlevelfeature)
        residual_features = residuals_phones(base_features_train, base_features_val, stimulus_features_train, stimulus_features_test)
    
    return residual_features
        
def load_model(model_path, subject, device):
    nc_mask, out_dim = get_voxels_nc_mask(subject)
    model = Wav2VecLinear(out_dim, 'wav2vec2-base-960h').eval().to(device)
    if model_path is not None:
        print(f'loading ckpt {model_path}')
        ckpt = torch.load(model_path, map_location=device)
        if 'module.' in list(ckpt.keys())[0]:
            ## for data parallel model
            ckpt = {key.replace('module.', ''): value for key, value in ckpt.items()}
            
        model.load_state_dict(ckpt)  # Load the model
    model.eval()
    return model


def load_model_wembed(model_path, device):
    out_dim = 25600
    model = Wav2VecLinear(out_dim).eval().to(device)
    if model_path is not None:
        print(f'loading ckpt {model_path}')
        ckpt = torch.load(model_path, map_location=device)
        if 'module.' in list(ckpt.keys())[0]:
            ## for data parallel model
            ckpt = {key.replace('module.', ''): value for key, value in ckpt.items()}
            
        model.load_state_dict(ckpt)  # Load the model
    model.eval()
    return model

def get_wav_features(model,
                     story_name,
                     subject,
                     chunksz=100,
                     contextsz=16000,
                     layers = [2, 5, 7, 8, 10, 12],
                     ):
    # Load the model

    wav_tensor, fmri_tensor = load_story_data(story_name, subject)
    # Extract speech features
    chunksz_sec = chunksz / 1000.

    # context size in terms of chunks
    assert (contextsz % chunksz) == 0, "These must be divisible"
    contextsz_sec = contextsz / 1000.
    
    model_config =  {
        "huggingface_hub": "facebook/wav2vec2-base-960h",
        "stride": 320,
        "min_input_length": 400
    }
    extract_features_kwargs = {'model': model.wav2vec, 'feature_extractor': model.processor, 'model_config': model_config,
        'wav': wav_tensor, 'sel_layers': layers,
        'chunksz_sec': chunksz_sec, 'contextsz_sec': contextsz_sec,
        'require_full_context': False,
        'batchsz': 1, 'return_numpy': False
    }
    
    
    wav_features = extract_speech_features(**extract_features_kwargs) 
    return wav_features, fmri_tensor

def predict_stories(subject,
                    model_path=None,
                    lowlevelfeature='articulation',
                    layers=[2, 5, 7, 8, 10, 12],
                    model_pref='base',
                    save_dir='../outputs/preds_results',
                    batch_size=32,
                    device='cuda'):
    # Predict fMRI
    if model_path is None:
        model_suff = f'base_wav2vec_subj{subject}'
    else:
        model_suff =  model_path.split('/')[-2] +  '_' + model_path.split('/')[-1].replace('.pth', '')[0]
    
    if 'embed' in model_path.split('/')[-2]:
        model_suff = f"{model_suff}_whisper_embed"
        
    register_path = f'../datasets/cached_preds/{model_suff}_subj{subject}/'
    if os.path.exists(register_path):
        layers_outputs = pickle.load(open(f'{register_path}/layers_outputs.pkl', 'rb'))
        final_outputs = pickle.load(open(f'{register_path}/final_outputs.pkl', 'rb'))
        fmri_tensors = pickle.load(open(f'{register_path}/fmri_tensors.pkl', 'rb'))
        print('Loaded from cache')
    else:
        print('Computing features')
        if 'embed' in model_suff:
            print('wembed model')
            model = load_model_wembed(model_path, device)
        else:
            model = load_model(model_path, subject, device)
            
        layers_outputs = {}; final_outputs = {}; fmri_tensors = {}
        for story_name in tqdm(residual_train_stories + residual_test_stories):
            print(f'Processing {story_name}')
            wav_features, fmri_tensor = get_wav_features(model, story_name, subject, layers=layers, batch_size=batch_size, device=device)
            
            downsampled_features = lanczosinterp2D(wav_features['final_outputs'].cpu().numpy(), wav_features['times'].numpy()[:, 1], wordseqs[story_name].tr_times) # downsample features
            final_outputs[story_name] = downsampled_features; fmri_tensors[story_name] = fmri_tensor.numpy()
            
            for layer in layers:
                if layer not in layers_outputs:
                    layers_outputs[layer] = {story_name: lanczosinterp2D(wav_features['module_features'][f'layer.{layer}'].cpu().numpy(), wav_features['times'].numpy()[:, 1], wordseqs[story_name].tr_times) }
                
                layers_outputs[layer][story_name] =  lanczosinterp2D(wav_features['module_features'][f'layer.{layer}'].cpu().numpy(), wav_features['times'].numpy()[:, 1], wordseqs[story_name].tr_times)
    
        os.makedirs(register_path, exist_ok=True)
        with open(f'{register_path}/layers_outputs.pkl', 'wb') as f:
            pickle.dump(layers_outputs, f)
        with open(f'{register_path}/final_outputs.pkl', 'wb') as f:
            pickle.dump(final_outputs, f)
        with open(f'{register_path}/fmri_tensors.pkl', 'wb') as f:
            pickle.dump(fmri_tensors, f)
        
    f_stories = residual_train_stories + residual_test_stories
    final_outputs_train = np.nan_to_num(np.vstack([ridge_utils.npp.zs(final_outputs[story][10:-5]) for story in residual_train_stories]))
    final_outputs_test = np.nan_to_num(np.vstack([ridge_utils.npp.zs(final_outputs[story][10:-5]) for story in residual_test_stories]))
    # now predict from residuals
    final_layer_residuals = get_residuals(final_outputs_train, final_outputs_test, lowlevelfeature)
    pred_res_final = predict_encoding(final_layer_residuals, fmri_tensors, subject)
    os.makedirs(f'{save_dir}/UTS0{subject}', exist_ok=True)
    with open(f'{save_dir}/UTS0{subject}/{model_pref}_pred_res_final.pkl', 'wb') as f:
        pickle.dump(pred_res_final, f)
    
    for layer in layers:
        layer_outputs_train = np.nan_to_num(np.vstack([ridge_utils.npp.zs(layers_outputs[layer][story][10:-5]) for story in residual_train_stories]))
        layer_outputs_test = np.nan_to_num(np.vstack([ridge_utils.npp.zs(layers_outputs[layer][story][10:-5]) for story in residual_test_stories]))
        # now predict from residuals
        layer_residuals = get_residuals(layer_outputs_train, layer_outputs_test, lowlevelfeature)
        pred_res_layer = predict_encoding(layer_residuals, fmri_tensors, subject)
        with open(f'{save_dir}/UTS0{subject}/{model_pref}_pred_res_layer_{layer}.pkl', 'wb') as f:
            pickle.dump(pred_res_layer, f)
    



def predict_encoding(features, fmri_tensors, subject, trim_start=50, trim_end=5): 
        
    #Training data
    Rstim = features[0] 
    #Test data
    Pstim = features[1][40:] 

    # Add FIR delays
    delRstim = make_delayed(Rstim, delays)
    delPstim = make_delayed(Pstim, delays)
    alphas = np.logspace(1, 4, 15) # Equally log-spaced ridge parameters
    nboots = 3 # Number of cross-validation ridge regression runs

    # # Get response data
    Rresp = np.vstack([fmri_tensors[story] for story in residual_train_stories])
    Presp = np.vstack([fmri_tensors[story][40:] for story in residual_test_stories])
    print(f'stim data shapes: {delRstim.shape}, {delPstim.shape}, fmri data shapes: {Rresp.shape}, {Presp.shape}')

    # Bootstrap chunking parameters
    chunklen = 20
    nchunks = int(len(Rresp) * 0.25 / chunklen)

    # Run ridge regression 
    wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(delRstim, Rresp, delPstim, Presp,
                                                        alphas, nboots, chunklen, nchunks,
                                                        use_corr=False, single_alpha=False)
    
    pred = np.dot(delPstim,  wt)
    dtest = load_fmri_story(test_stories[0], subject)
    
    SPE, cc_norm, cc_max, corrs_unnorm = spe_and_cc_norm(dtest['individual_repeats'][:, 40:, :], pred, max_flooring=0.25) # NC via computing the SPE and CC
    res_dict = dict(corr=corr, c_norm=cc_norm, cc_max=cc_max, corrs_unnorm=corrs_unnorm,)
    
    
    return res_dict
    



if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description = "Eval Residuals argparser")
    parser.add_argument("--ft", help="fine-tuned vs pretrained", action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for eval')
    parser.add_argument('--model_path', type=str, default='', help='path of saved model')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    
    args = parser.parse_args()

    base_test = True
    ft = args.ft
    device = args.device


    for subj in SUBJ_LIST:
        subject = subj; 
        for lowlevelfeature in ['powspec', 'diphone', 'triphone', 'articulation']:

            print(f'processing subj {subj}, lowlevelfeature {lowlevelfeature}')
                
            if ft:
                march = f'residual_wav2vec_btuned_{lowlevelfeature}_{subject}' 
            else:
                march = f'residual_wav2vec_base_{lowlevelfeature}_{subject}'
            model_pref = f"mp_{march}"
            
            if ft:
                predict_stories(subject=subject, model_path=args.model_path, lowlevelfeature=lowlevelfeature, model_pref=model_pref, batch_size=args.batch_size, device=device)
                
            else:
                predict_stories(subject=subject, model_path=None, lowlevelfeature=lowlevelfeature, model_pref=model_pref, batch_size=args.batch_size, device=device)
