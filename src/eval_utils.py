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
from transformers import Wav2Vec2Model, Wav2Vec2Processor, PreTrainedModel
import cortex
# from extract_speech_features import extract_speech_features
### Some extra helper functions

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


grids = joblib.load("../datasets/story_data/grids_huge.jbl") # Load TextGrids containing story annotations
trfiles = joblib.load("../datasets/story_data/trfiles_huge.jbl") # Load TRFiles containing TR information

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
    sub_nc = np.load(f'../datasets/subject_NCs/UTS0{subject}.npy')
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
    sub_nc = np.load(f'../datasets/subject_NCs/UTS0{subject}.npy')
    
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
    
    orig_data: 3D numpy array (trials, timepoints, voxels)
    
    data_pred: 2D numpy array (timepoints, voxels)
    
    data_norm: bool -> Set to False if not pre-normalized
    
    max_flooring: None/float (0-1) -> If not None, compute cc_norm in an alternate way that floors cc_max by max_flooring.
    This is helpful to clean up bad voxels that are not at all language selective.
    
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

def extract_speech_features(model: PreTrainedModel, model_config: dict, wav: torch.Tensor,
                            chunksz_sec: float, contextsz_sec: float,
                            num_sel_frames = 1, frame_skip = 5, sel_layers: Optional[List[int]]=None,
                            batchsz: int = 1,
                            return_numpy: bool = True, move_to_cpu: bool = True,
                            disable_tqdm: bool = False, feature_extractor=None,
                            sampling_rate: int = TARGET_SAMPLE_RATE, require_full_context: bool = False,
                            stereo: bool = False):
    assert (num_sel_frames == 1), f"'num_sel_frames` must be 1 to ensure causal feature extraction, but got {num_sel_frames}. "\
        "This option will be deprecated in the future."
    if stereo:
        raise NotImplementedError("stereo not implemented")
    else:
        assert wav.ndim == 1, f"input `wav` must be 1-D but got {wav.ndim}"
    if return_numpy: assert move_to_cpu, "'move_to_cpu' must be true if returning numpy arrays"

    model.eval()
    # Compute chunks & context sizes in terms of samples & context
    chunksz_samples = int(chunksz_sec * sampling_rate)
    contextsz_samples = int(contextsz_sec * sampling_rate)

    # `snippet_ends` has the last (exclusive) sample for each snippet
    snippet_ends = []
    if not require_full_context:
        # Add all snippets that are _less_ than the total input size
        # (context+chunk)
        snippet_ends.append(torch.arange(chunksz_samples, contextsz_samples+chunksz_samples, chunksz_samples))

    # Add all snippets that are exactly the length of the requested input
    # (`Tensor.unfold` is basically a sliding window).
    if wav.shape[0] >= chunksz_samples+contextsz_samples:
        # `unfold` fails if `wav.shape[0]` is less than the window size.
        snippet_ends.append(
            torch.arange(wav.shape[0]).unfold(0, chunksz_samples+contextsz_samples, chunksz_samples)[:,-1]+1
        )

    snippet_ends = torch.cat(snippet_ends, dim=0) # shape: (num_snippets,)

    if snippet_ends.shape[0] == 0:
        raise ValueError(f"No snippets possible! Stimulus is probably too short ({wav.shape[0]} samples). Consider reducing context size or setting `require_full_context=True`")

    # 2-D array where `[i,0]` and `[i,1]` are the start and end, respectively,
    # of snippet `i` in samples. Shape: (num_snippets, 2)
    snippet_times = torch.stack([torch.maximum(torch.zeros_like(snippet_ends),
                                               snippet_ends-(contextsz_samples+chunksz_samples)),
                                 snippet_ends], dim=1)

    # Remove snippets that are not long enough. (Seems easier to filter
    # after generating the snippet bounds than handling it above in each case)
    # TODO: is there any way to programatically check this in HuggingFace?
    # doesn't seem so (unlike s3prl).
    if 'min_input_length' in model_config:
        # this is stored originally in **samples**!!!
        min_length_samples = model_config['min_input_length']
    elif 'win_ms' in model.config:
        min_length_samples = model.config['win_ms'] / 1000. * TARGET_SAMPLE_RATE

    snippet_times = snippet_times[(snippet_times[:,1] - snippet_times[:,0]) >= min_length_samples]
    snippet_times_sec = snippet_times / sampling_rate # snippet_times, but in sec.

    module_features = collections.defaultdict(list)
    out_features = [] # the final output of the model
    times = [] # times are shared across all layers

    #assert (frames_per_chunk % frame_skip) == 0, "These must be divisible"
    frame_len_sec = model_config['stride'] / TARGET_SAMPLE_RATE # length of an output frame (sec.)

    snippet_length_samples = snippet_times[:,1] - snippet_times[:,0] # shape: (num_snippets,)
    if require_full_context:
        assert all(snippet_length_samples == snippet_length_samples[0]), "uneven snippet lengths!"
        snippet_length_samples = snippet_length_samples[0]
        assert snippet_length_samples.ndim == 0

    # Set up the iterator over batches of snippets
    if require_full_context:
        # This case is simpler, so handle it explicitly
        snippet_batches = snippet_times.T.split(batchsz, dim=1)
    else:
        # First, batch the snippets that are of different lengths.
        snippet_batches = snippet_times.tensor_split(torch.where(snippet_length_samples.diff() != 0)[0]+1, dim=0)
        # Then, split any batches that are too big to fit into the given
        # batch size.
        snippet_iter = []
        for batch in snippet_batches:
            # split, *then* transpose
            if batch.shape[0] > batchsz:
                snippet_iter += batch.T.split(batchsz,dim=1)
            else:
                snippet_iter += [batch.T]
        snippet_batches = snippet_iter

    snippet_iter = snippet_batches
    if not disable_tqdm:
        snippet_iter = tqdm(snippet_iter, desc='snippet batches', leave=False)
    snippet_iter = enumerate(snippet_iter)


    # Iterate with a sliding window. stride = chunk_sz
    for batch_idx, (snippet_starts, snippet_ends) in snippet_iter:
        if ((snippet_ends - snippet_starts) < (contextsz_samples + chunksz_samples)).any() and require_full_context:
            raise ValueError("This shouldn't happen with require_full_context")

        # If we don't have enough samples, skip this chunk.
        if (snippet_ends - snippet_starts < min_length_samples).any():
            print('If this is true for any, then you might be losing more snippets than just the offending (too short) snippet')
            assert False

        # Construct the input waveforms for the batch
        batched_wav_in_list = []
        for batch_snippet_idx, (snippet_start, snippet_end) in enumerate(zip(snippet_starts, snippet_ends)):
            # Stacking might be inefficient, so populate a pre-allocated array.
            #batched_wav_in[batch_snippet_idx, :] = wav[snippet_start:snippet_end]
            # But stacking makes variable batch size easier!
            batched_wav_in_list.append(wav[snippet_start:snippet_end])
        batched_wav_in = torch.stack(batched_wav_in_list, dim=0)

        # The final batch may be incomplete if batchsz doesn't evenly divide
        # the number of snippets.
        if (snippet_starts.shape[0] != batched_wav_in.shape[0]) and (snippet_starts.shape[0] != batchsz):
            batched_wav_in = batched_wav_in[:snippet_starts.shape[0]]

        # Take the last 1 or 2 activations, and time-wise put it at the
        # end of chunk.
        output_inds = np.array([-1 - frame_skip*i for i in reversed(range(num_sel_frames))])

        # Use a pre-processor if given (e.g. to normalize the waveform), and
        # then feed into the model.
        if feature_extractor is not None:
            # This step seems to be NOT differentiable, since the feature
            # extractor first converts the Tensor to a numpy array, then back
            # into a Tensor.
            # If you want to backprop through the stimulus, you might have to
            # re-implement the feature extraction in PyTorch (in particular, the
            # normalization)

            if stereo: raise NotImplementedError("Support handling multi-channel audio with feature extractor")
            # It looks like most feature extractors (e.g.
            # Wav2Vec2FeatureExtractor) accept mono audio (i.e. 1-dimensional),
            # but it's unclear if they support stereo as well.

            feature_extractor_kwargs = {}

            features_key = 'input_values'

            preprocessed_snippets = feature_extractor(list(batched_wav_in.cpu().numpy()),
                                                      return_tensors='pt',
                                                      sampling_rate=sampling_rate,
                                                      **feature_extractor_kwargs)

                # sampling rates must match if not using a pre-processor
            assert sampling_rate == TARGET_SAMPLE_RATE, f"sampling rate mismatch! {sampling_rate} != {TARGET_SAMPLE_RATE}"

            with torch.no_grad():
                chunk_features = model(preprocessed_snippets[features_key].to(model.device), output_hidden_states=True)
        else:
            with torch.no_grad():
                chunk_features = model(batched_wav_in, output_hidden_states=True)

        # Make sure we have enough outputs
        if(chunk_features['last_hidden_state'].shape[1] < (num_sel_frames-1) * frame_skip - 1):
            print("Skipping:", snippet_idx, "only had", chunk_features['last_hidden_state'].shape[1],
                    "outputs, whereas", (num_sel_frames-1) * frame_skip - 1, "were needed.")
            continue

        assert len(output_inds) == 1, "Only one output per evaluation is "\
            "supported for Hugging Face (because they don't provide the downsampling rate)"



        for out_idx, output_offset in enumerate(output_inds):
            times.append(torch.stack([snippet_starts, snippet_ends], dim=1))
            # print(chunk_features)
            output_representation = chunk_features['last_hidden_state'][:, output_offset, :] # shape: (batchsz, hidden_size)
            if move_to_cpu: output_representation = output_representation.cpu()
            if return_numpy: output_representation = output_representation.numpy()
            out_features.append(output_representation)

            # Collect features from individual layers
            # NOTE: outs['hidden_states'] might have an extra element at
            # the beginning for the feature extractor.
            # e.g. 25 "layers" --> CNN output + 24 transformer layers' output
            for layer_idx, layer_activations in enumerate(chunk_features['hidden_states']):
                # Only save layers that the user wants (if specified)
                if sel_layers:
                    if layer_idx not in sel_layers: continue

                layer_representation = layer_activations[:, output_offset, :] # shape: (batchsz, hidden_size)
                if move_to_cpu: layer_representation = layer_representation.cpu()
              
                module_name = f"layer.{layer_idx}"

                module_features[module_name].append(layer_representation)

    out_features = np.concatenate(out_features, axis=0) if return_numpy else torch.cat(out_features, dim=0) # shape: (timesteps, features)
    module_features = {name: (np.concatenate(features, axis=0) if return_numpy else torch.cat(features, dim=0))\
                       for name, features in module_features.items()}

    assert all(features.shape[0] == out_features.shape[0] for features in module_features.values()),\
        "Missing timesteps in the module activations!! (possible PyTorch bug)"
    times = torch.cat(times, dim=0) / TARGET_SAMPLE_RATE # convert samples --> seconds. shape: (timesteps,)
    if return_numpy: times = times.numpy()

    del chunk_features # possible memory leak. remove if unneeded
    return {'final_outputs': out_features, 'times': times,
            'module_features': module_features}
