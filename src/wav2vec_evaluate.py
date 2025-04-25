import numpy as np  
import cortex 
import torch 
import matplotlib.pyplot as plt 
from extract_speech_features import extract_speech_features
# from wav2vec_sim import Wav2VecLinear 
from wav2vec_linear import Wav2VecLinear, Wav2VecLoRA, Wav2VecLoRAFilLayer, Wav2VecLoRAEarly

from transformers import AutoModel, AutoModelForPreTraining, PreTrainedModel, AutoFeatureExtractor
from eval_utils import *
from ridge_utils.interpdata import lanczosinterp2D
import ridge_utils.npp
from ridge_utils.ridge import bootstrap_ridge
from ridge_utils.util import make_delayed
import pickle
from glob import glob
from argparse import ArgumentParser

from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Processor
def load_ssl_model(model_path):
    model = Wav2Vec2ForPreTraining.from_pretrained(model_path)
    model.eval()
    model.processor = Wav2Vec2Processor.from_pretrained(f"facebook/wav2vec2-base", return_tensors="pt")
    return model

def get_lora_keys(state_dict):
    lora_state_dict = {}
    for key in state_dict.keys():
        if 'lora_model' in key:
            lkey = key.replace('module.lora_model.', '')
            lora_state_dict[lkey] = state_dict[key]
    return lora_state_dict

def load_lora_model(model_path, subject, device):

    nc_mask, out_dim = get_voxels_nc_mask(subject)
    model = Wav2VecLoRA(out_dim, lora_rank=8).to(device)
    
    if model_path is not None:
        print(f'loading LoRA ckpt {model_path}')
        ckpt = torch.load(model_path, map_location=device)
        lora_dict = get_lora_keys(ckpt)
        lora_model = model.lora_model
        lora_model.load_state_dict(lora_dict)
            
        # model.load_state_dict(ckpt)  # Load the model
    lora_model.eval()
    model_config = model.wav2vec.config
    processor = model.processor
    
    return lora_model, processor, model_config, nc_mask
    
def load_model(model_path, subject, device):
    nc_mask, out_dim = get_voxels_nc_mask(subject)
    model = Wav2VecLinear(out_dim, 'facebook/wav2vec2-base-960h').eval().to(device)
    if model_path is not None:
        print(f'loading ckpt {model_path}')
        ckpt = torch.load(model_path, map_location=device)
        ckpt_w = {}
        for k in ckpt:
            if k not in ['module.linear.weight', 'module.linear.bias']:
                ckpt_w[k.replace('module.wav2vec.', '')] = ckpt[k]
    
        ckpt_w["encoder.pos_conv_embed.conv.weight_g"] = ckpt_w['encoder.pos_conv_embed.conv.parametrizations.weight.original0']
        ckpt_w["encoder.pos_conv_embed.conv.weight_v"] = ckpt_w['encoder.pos_conv_embed.conv.parametrizations.weight.original1']
        ckpt_w.pop('encoder.pos_conv_embed.conv.parametrizations.weight.original0'); ckpt_w.pop('encoder.pos_conv_embed.conv.parametrizations.weight.original1')

        
        model.wav2vec.load_state_dict(ckpt_w)  # Load the model
    model.eval()
    model_w = model.wav2vec
    model_w.eval()
    model_config = model.wav2vec.config
    processor = model.processor
    # model = load_ssl_model().eval().to(device)
    return model_w, processor, model_config, nc_mask

def load_model_wembed(model_path, device):
    out_dim = 5376
    model = Wav2VecLinear(out_dim).eval().to(device)
    if model_path is not None:
        print(f'loading ckpt {model_path}')
        ckpt = torch.load(model_path, map_location=device)
        # if 'module.' in list(ckpt.keys())[0]:
        #     ## for data parallel model
        #     ckpt = {key.replace('module.', ''): value for key, value in ckpt.items()}
        ckpt_w = {}
        for k in ckpt:
            if k not in ['module.linear.weight', 'module.linear.bias']:
                ckpt_w[k.replace('module.wav2vec.', '')] = ckpt[k]
    
        ckpt_w["encoder.pos_conv_embed.conv.weight_g"] = ckpt_w['encoder.pos_conv_embed.conv.parametrizations.weight.original0']
        ckpt_w["encoder.pos_conv_embed.conv.weight_v"] = ckpt_w['encoder.pos_conv_embed.conv.parametrizations.weight.original1']
        ckpt_w.pop('encoder.pos_conv_embed.conv.parametrizations.weight.original0'); ckpt_w.pop('encoder.pos_conv_embed.conv.parametrizations.weight.original1')

        
        model.wav2vec.load_state_dict(ckpt_w)  # Load the model
    model.eval()
    return model

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
    # Load the model
    # model_path = f'train_logs/wav2vec_story_subj{subject}_mean_epochs_30.pth'
    # Load the story data
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
    
    #TODO : accept model only without .wav2vec
    extract_features_kwargs = {'model': model, 'feature_extractor': feature_extractor, 'model_config': model_config,
        'wav': wav_tensor, 'sel_layers': layers,
        'chunksz_sec': chunksz_sec, 'contextsz_sec': contextsz_sec,
        'require_full_context': False,
        'batchsz': 1, 'return_numpy': False
    }
    
    
    wav_features = extract_speech_features(**extract_features_kwargs) # keys {'final_outputs': out_features, 'times': times: 'module_features': module_features}
    print(wav_features['final_outputs'].shape)
    # print(wav_features['module_features'])
    return wav_features, fmri_tensor

def predict_stories(subject, model_path=None, layers=[2, 5, 7, 8, 10, 11, 12], model_pref='base', save_dir='../outputs/preds_results', batch_size=32, device='cuda'):
    # model_path = f'train_logs/wav2vec_story_subj{subject}_mean_epochs_30.pth'
    # Predict fMRI
    if model_path is not None:
        if 'wembed' in model_path.split('/')[-2]: #TODO: don't need it anymore cause out_dim doesnt matter?
            print('wembed model')
            model, processor, model_config, nc_mask = load_model_wembed(model_path, device)
        elif 'lora' in model_path.split('/')[-2]:
            #TODO: repeat for other cases to get all model configs in load fun
            # model, nc_mask = load_lora_model(model_path, subject, device)
            model, processor, model_config, nc_mask = load_lora_model(model_path, subject, device)
        else:
           model, processor, model_config, nc_mask = load_model(model_path, subject, device)
    else:
        model, processor, model_config, nc_mask = load_model(model_path, subject, device)
        
        
        
    layers_outputs = {}; final_outputs = {}; fmri_tensors = {}
    for story_name in tqdm(train_stories + test_stories):
        print(f'Processing {story_name}')
        wav_features, fmri_tensor = get_wav_features(model, processor, model_config, story_name, subject, layers=layers, batch_size=batch_size, device=device)
        
        downsampled_features = lanczosinterp2D(wav_features['final_outputs'].cpu().numpy(), wav_features['times'].numpy()[:, 1], wordseqs[story_name].tr_times) # downsample features
        final_outputs[story_name] = downsampled_features; fmri_tensors[story_name] = fmri_tensor.numpy()
        
        for layer in layers:
            if layer not in layers_outputs:
                layers_outputs[layer] = {story_name: lanczosinterp2D(wav_features['module_features'][f'layer.{layer}'].cpu().numpy(), wav_features['times'].numpy()[:, 1], wordseqs[story_name].tr_times) }
            
            layers_outputs[layer][story_name] =  lanczosinterp2D(wav_features['module_features'][f'layer.{layer}'].cpu().numpy(), wav_features['times'].numpy()[:, 1], wordseqs[story_name].tr_times)
    ## 
    # f'UTS0{subject}
    pred_res_final = predict_encoding(final_outputs, fmri_tensors, subject)
    os.makedirs(f'{save_dir}/UTS0{subject}', exist_ok=True)
    with open(f'{save_dir}/UTS0{subject}/{model_pref}_pred_res_final.pkl', 'wb') as f:
        pickle.dump(pred_res_final, f)
    
    for layer in layers:
        pred_res_layer = predict_encoding(layers_outputs[layer], fmri_tensors, subject)
        with open(f'{save_dir}/UTS0{subject}/{model_pref}_pred_res_layer_{layer}.pkl', 'wb') as f:
            pickle.dump(pred_res_layer, f)
    
    # return



def predict_encoding(features, fmri_tensors, subject, trim_start=50, trim_end=5): 
        
    #Training data
    Rstim = np.nan_to_num(np.vstack([ridge_utils.npp.zs(features[story][10:-5]) for story in train_stories]))

    #Test data
    Pstim = np.nan_to_num(np.vstack([ridge_utils.npp.zs(features[story][trim_start:-trim_end]) for story in test_stories]))

    # Add FIR delays
    delRstim = make_delayed(Rstim, delays)
    delPstim = make_delayed(Pstim, delays)
    alphas = np.logspace(1, 4, 15) # Equally log-spaced ridge parameters between 10 and 10000. 
    nboots = 3 # Number of cross-validation ridge regression runs. You can lower this number to increase speed.

    # # Get response data
    # resp_dict = joblib.load("responses/full_responses/UTS01_responses.jbl") # Located in story_responses folder
    Rresp = np.vstack([fmri_tensors[story] for story in train_stories])
    Presp = np.vstack([fmri_tensors[story][40:] for story in test_stories])
    print(f'stim data shapes: {delRstim.shape}, {delPstim.shape}, fmri data shapes: {Rresp.shape}, {Presp.shape}')

    # Bootstrap chunking parameters
    chunklen = 20
    nchunks = int(len(Rresp) * 0.25 / chunklen)

    # Run ridge regression - this might take some time
    wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(delRstim, Rresp, delPstim, Presp,
                                                        alphas, nboots, chunklen, nchunks,
                                                        use_corr=False, single_alpha=False)
    
    pred = np.dot(delPstim,  wt)
    dtest = load_fmri_story(test_stories[0], subject)
    
    SPE, cc_norm, cc_max, corrs_unnorm = spe_and_cc_norm(dtest['individual_repeats'][:, 40:, :], pred, max_flooring=0.25) # NC via computing the SPE and CC
    # res_dict = dict(pred=pred, wt=wt, corr=corr, c_norm=cc_norm, cc_max=cc_max, corrs_unnorm=corrs_unnorm,)
    res_dict = dict(corr=corr, c_norm=cc_norm, cc_max=cc_max, corrs_unnorm=corrs_unnorm,)
    
    
    return res_dict
    
    
if __name__=='__main__':

    parser = ArgumentParser()
    parser.add_argument('--subject', type=int, default=8, help='Subject number')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--logs_dir', type=str, default='./logs')
    parser.add_argument('--base_test', action='store_true', help='Whether to test the base model')
    args = parser.parse_args()
    
    
    subject = args.subject
    device = args.device
    base_test = args.base_test
    
    model_paths = glob(f'{args.logs_dir}/*')
    model_paths = (sorted(model_paths, key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]), reverse=True))

    
    if base_test:
        march = f"wav2vec_story_base_subj_{subject}" 
        model_pref = f"mp_{march}"
        predict_stories(subject=subject, model_path=None, model_pref=model_pref, batch_size=256, device=device)
    
    else:
        desired_epochs = [ 3, 5, 10, 20, 30] 
        
        for model_path in model_paths:
            epoch = int(model_path.split('/')[-1].split('_')[-1].split('.')[0])
            
            if epoch not in desired_epochs:
                continue
            print(f'Processing {model_path}')
            march = f'wav2vec_story_mp' 
            model_pref = f"mp_{march}_epoch_{epoch}_subj_{subject}"
            predict_stories(subject=subject, model_path=model_path, model_pref=model_pref, batch_size=1024, device=device, save_dir='../outputs/preds_results')
    
        
