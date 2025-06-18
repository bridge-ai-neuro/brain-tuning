import numpy as np  
import cortex 
import torch 
import matplotlib.pyplot as plt 
from extract_speech_features import extract_speech_features
from wav2vec_linear import Wav2VecLinear
from hubert_linear import HubertLinear
from whisper_linear import WhisperLinear

from eval_utils import *
from ridge_utils.interpdata import lanczosinterp2D
import ridge_utils.npp
from ridge_utils.ridge import bootstrap_ridge
from ridge_utils.util import make_delayed
import pickle
from glob import glob
from argparse import ArgumentParser

def init_model(out_dim, device):
    
    if 'wav2vec' in model_name:
        model = Wav2VecLinear(out_dim=out_dim,
                              model_name=model_name, # default is 'facebook/wav2vec2-base',
                              ).wav2vec
        save_key = 'wav2vec'
        processor = model.processor
        config = model.wav2vec.config
    elif 'hubert' in model_name:
        model = HubertLinear(out_dim=out_dim,
                             model_name=model_name, # default is 'facebook/hubert-base-ls960',
                             ).hubert
        save_key = 'hubert'
        processor = model.processor
        config = model.hubert.config
    elif 'whisper' in model_name:
        model = WhisperLinear(out_dim=out_dim,
                              model_name=model_name, # default is 'openai/whisper-small',
                              ).whisper_encoder
        save_key = 'whisper_encoder'
        processor = model.feature_extractor
        config = model.whisper_model.config
    else:
        raise ValueError(f"Model {model_name} is not supported. Choose from 'wav2vec', 'hubert', or 'whisper'.")
    return model.eval().to(device), save_key, processor, config

def load_model(model_path, subject, device):
    nc_mask, out_dim = get_voxels_nc_mask(subject)
    model, save_key, processor, model_config = init_model(out_dim=out_dim, device=device)
    if model_path is not None:
        print(f'loading ckpt {model_path}')
        ckpt = torch.load(model_path, map_location=device)
        ckpt_w = {}
        for k in ckpt:
            if k not in ['module.linear.weight', 'module.linear.bias']:
                ckpt_w[k.replace(f'module.{save_key}.', '')] = ckpt[k]
        
        model.load_state_dict(ckpt_w)  # Load the brain-tuned model weights
        
    model.eval()
    processor = model.processor
    return model, processor, model_config, nc_mask



def get_wav_features(model,
                     feature_extractor,
                     model_config,
                     story_name,
                     subject,
                     chunksz=100,
                     contextsz=16000,
                     layers = [2, 5, 7, 8, 10, 11, 12],
                     batch_size=32,
                     device='cuda'):

    wav_tensor, fmri_tensor = load_story_data(story_name, subject)
    chunksz_sec = chunksz / 1000.

    assert (contextsz % chunksz) == 0, "These must be divisible"
    contextsz_sec = contextsz / 1000.
    
    model_config =  {
        "huggingface_hub": model_name,
        "stride": 320,
        "min_input_length": 400
    }
    
    extract_features_kwargs = {'model': model, 'feature_extractor': feature_extractor, 'model_config': model_config,
        'wav': wav_tensor, 'sel_layers': layers,
        'chunksz_sec': chunksz_sec, 'contextsz_sec': contextsz_sec,
        'require_full_context': False,
        'batchsz': batch_size, 'return_numpy': False, 'is_whisper' : 'whisper' in model_name
    }
    
    
    wav_features = extract_speech_features(**extract_features_kwargs) # keys {'final_outputs': out_features, 'times': times: 'module_features': module_features}
    print(wav_features['final_outputs'].shape)

    return wav_features, fmri_tensor

def predict_encoding(features, fmri_tensors, subject, trim_start=50, trim_end=5): 
        
    #Training data
    Rstim = np.nan_to_num(np.vstack([ridge_utils.npp.zs(features[story][10:-5]) for story in train_stories]))

    #Test data
    Pstim = np.nan_to_num(np.vstack([ridge_utils.npp.zs(features[story][trim_start:-trim_end]) for story in test_stories]))

    # Add FIR delays
    delRstim = make_delayed(Rstim, delays)
    delPstim = make_delayed(Pstim, delays)
    alphas = np.logspace(1, 4, 15) # Equally log-spaced ridge parameters between 10 and 10000. 
    nboots = 3 # Number of cross-validation ridge regression runs.

    Rresp = np.vstack([fmri_tensors[story] for story in train_stories])
    Presp = np.vstack([fmri_tensors[story][40:] for story in test_stories])
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
    
    SPE, cc_norm, cc_max, corrs_unnorm = spe_and_cc_norm(dtest['individual_repeats'][:, 40:, :],
                                                         pred, max_flooring=0.25) # NC via computing the SPE and CC
    
    res_dict = dict(corr=corr, c_norm=cc_norm, cc_max=cc_max, corrs_unnorm=corrs_unnorm,)

    return res_dict

def predict_stories(subject,
                    model_path=None,
                    layers=[2, 5, 7, 8, 10, 11, 12],
                    model_pref='base',
                    save_dir='../outputs/brain_eval_results',
                    batch_size=32,
                    device='cuda'):


    model, processor, model_config, nc_mask = load_model(model_path, subject, device)
        
        
    layers_outputs = {}; final_outputs = {}; fmri_tensors = {}
    for story_name in tqdm(train_stories + test_stories):
        print(f'Processing {story_name}')
        wav_features, fmri_tensor = get_wav_features(model,
                                                     processor,
                                                     model_config,
                                                     story_name,
                                                     subject,
                                                     layers=layers,
                                                     batch_size=batch_size,
                                                     device=device)
        
        downsampled_features = lanczosinterp2D(wav_features['final_outputs'].cpu().numpy(),
                                               wav_features['times'].numpy()[:, 1],
                                               wordseqs[story_name].tr_times) # downsample features
        
        final_outputs[story_name] = downsampled_features; fmri_tensors[story_name] = fmri_tensor.numpy()
        
        for layer in layers:
            if layer not in layers_outputs:
                layers_outputs[layer] = {story_name: lanczosinterp2D(wav_features['module_features'][f'layer.{layer}'].cpu().numpy(),
                                                                     wav_features['times'].numpy()[:, 1],
                                                                     wordseqs[story_name].tr_times) }
            
            layers_outputs[layer][story_name] = lanczosinterp2D(wav_features['module_features'][f'layer.{layer}'].cpu().numpy(),
                                                                 wav_features['times'].numpy()[:, 1],
                                                                 wordseqs[story_name].tr_times)

    pred_res_final = predict_encoding(final_outputs, fmri_tensors, subject)
    os.makedirs(f'{save_dir}/UTS0{subject}', exist_ok=True)
    with open(f'{save_dir}/UTS0{subject}/{model_pref}_pred_res_final.pkl', 'wb') as f:
        pickle.dump(pred_res_final, f)
    
    for layer in layers:
        pred_res_layer = predict_encoding(layers_outputs[layer], fmri_tensors, subject)
        with open(f'{save_dir}/UTS0{subject}/{model_pref}_pred_res_layer_{layer}.pkl', 'wb') as f:
            pickle.dump(pred_res_layer, f)
    




    
    
if __name__=='__main__':

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='facebook/wav2vec2-base', help='Name of the pre-trained model')
    parser.add_argument('--subject', type=int, default=3, help='Subject number')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--logs_dir', type=str, default='../outputs/train_logs/wav2vec_story_mp')
    parser.add_argument('--save_dir', type=str, default='../outputs/brain_eval_results')
    
    parser.add_argument('--base_test', action='store_true', help='Whether to test the pretrained model')
    args = parser.parse_args()
    
    model_name = args.model_name
    subject = args.subject
    device = args.device
    base_test = args.base_test
    os.makedirs(args.save_dir, exist_ok=True)
    
    model_paths = glob(f'{args.logs_dir}/*')
    model_paths = (sorted(model_paths, key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]), reverse=False))

    
    if base_test: # evaluate pretrained model
        march = f"eval_base_subj_{subject}" 
        model_pref = f"mp_{march}"
        predict_stories(subject=subject, model_path=None, model_pref=model_pref, batch_size=1024, device=device)
    
    else: # for brain-tuned models, evaluate multiple epochs
        desired_epochs = [ 5, 10, 20] 
        
        for model_path in model_paths:
            epoch = int(model_path.split('/')[-1].split('_')[-1].split('.')[0])
            
            if epoch not in desired_epochs:
                continue
            print(f'Processing {model_path}')
            march = f'story_mp' 
            model_pref = f"{march}_epoch_{epoch}_subj_{subject}"
            predict_stories(subject=subject, model_path=model_path, model_pref=model_pref, batch_size=1024, device=device, save_dir=args.save_dir)
    
        
