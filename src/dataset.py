import torch 
from torch.utils.data import Dataset
import os
import json
import numpy as np
import random 
import h5py
import random
ndelays = 4 # We use 4 FIR delays (2 seconds, 4 seconds, 6 seconds, 8 seconds)
delays = range(0, ndelays + 1)
## set seed for random exp
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays] 
    (in samples).
    
    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt,ndim = stim.shape
    dstims = []
    for di,d in enumerate(reversed(delays)):
        dstim = torch.zeros((nt, ndim))
        if d<0: ## negative delay
            dstim[:d,:] = stim[-d:,:]
            if circpad:
                dstim[d:,:] = stim[:-d,:]
        elif d>0:
            dstim[d:,:] = stim[:-d,:]
            if circpad:
                dstim[:d,:] = stim[-d:,:]
        else: ## d==0
            dstim = stim.clone()
        dstims.append(dstim)
    return torch.hstack(dstims)

TARGET_SAMPLE_RATE = 16000

def extract_speech_times(   
                        wav: torch.Tensor,
                        chunksz_sec: float=1, contextsz_sec: float=1,
                        num_sel_frames = 1,
                        sampling_rate: int = 16000, require_full_context: bool = False,
                        stereo: bool = False
                        ):
    assert (num_sel_frames == 1), f"'num_sel_frames` must be 1 to ensure causal feature extraction, but got {num_sel_frames}. "\
        "This option will be deprecated in the future."
    if stereo:
        raise NotImplementedError("stereo not implemented")
    else:
        assert wav.ndim == 1, f"input `wav` must be 1-D but got {wav.ndim}"

    # Compute chunks & context sizes in terms of samples & context
    chunksz_samples = int(chunksz_sec * sampling_rate)
    contextsz_samples = int(contextsz_sec * sampling_rate)

    # `snippet_ends` has the last (exclusive) sample for each snippet
    snippet_ends = []
    if not require_full_context:

        snippet_ends.append(torch.arange(chunksz_samples, contextsz_samples+chunksz_samples, chunksz_samples))

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
    
    times_in_seconds = snippet_times / sampling_rate
    return dict(
        indices=snippet_times,
        times=times_in_seconds
    )
    
def get_inds2tr_map(wav,
                    indices,
                    times,
                    tr_times,
                    used_chuncksz_sec = 1,
                    trim_start = 10,
                    trim_end = 5,
                    expectedtr=2.0045,):
    
    num_snippets = int(expectedtr//used_chuncksz_sec)
    aligned_wav = []
    fmri_inds = []


    for tridx, trtime in enumerate(tr_times[:]):

        if trtime < expectedtr:
            aligned_wav.append(torch.cat([torch.zeros(TARGET_SAMPLE_RATE), wav[:TARGET_SAMPLE_RATE]]))
            continue
        else:
            sidx, eidx = TARGET_SAMPLE_RATE * int(tr_times[tridx-1]), TARGET_SAMPLE_RATE * int(tr_times[tridx])
            if wav[sidx:eidx].shape[0] < TARGET_SAMPLE_RATE * 2:
                aligned_wav.append(torch.cat([torch.zeros(TARGET_SAMPLE_RATE * 2 - wav[sidx:eidx].shape[0]), wav[sidx:eidx]]))
            else:
                aligned_wav.append(wav[sidx:eidx])

    return torch.vstack(aligned_wav)


train_stories = list(np.load('../datasets/story_lists.npy')) #list(wordseqs.keys())
test_stories = ["wheretheressmoke"]
val_stories = ['stagefright']

train_stories_25 = list(['souls', 'howtodraw', 'alternateithicatom', 'hangtime', 'swimmingwithastronauts',
                                'eyespy', 'itsabox', 'myfirstdaywiththeyankees', 'legacy', 'theclosetthatateeverything',
                                'haveyoumethimyet', 'thatthingonmyarm', 'undertheinfluence', 'avatar',
                                'tildeath','naked', 'fromboyhoodtofatherhood', 'adollshouse',
                                'odetostepfather', 'adventuresinsayingyes', 'sloth', 'exorcism', 'buck', 'inamoment'])



residual_train_stories = ['alternateithicatom', 'avatar', 'howtodraw', 'legacy',
                'life', 'myfirstdaywiththeyankees', 'naked',
                'odetostepfather', 'souls', 'undertheinfluence']

residual_test_stories = ['wheretheressmoke']
    
for s in test_stories:
    if s in train_stories:
        train_stories.remove(s)
        
for s in val_stories:
    if s in train_stories:
        train_stories.remove(s)
        
## tr files info
import joblib
from ridge_utils.dsutils import make_word_ds
grids = joblib.load("../datasets/story_data/grids_huge.jbl") # Load TextGrids containing story annotations
trfiles = joblib.load("../datasets/story_data/trfiles_huge.jbl") # Load TRFiles containing TR information
wordseqs = make_word_ds(grids, trfiles)


class FMRIStory(Dataset):
    def __init__(self,
                 story_name,
                 subject,
                 sub_nc_mask,
                 read_stim_dir='../datasets/processed_stim_data_dp',
                 read_fmri_dir='../../ds003020/derivative/preprocessed_data/', #dataset downloaded from openneuro 
                 delays=range(0, ndelays + 1),
                 trim_start=10,
                 trim_end=5,
                 **wav_params):
        super(FMRIStory, self).__init__()
        self.story_name = story_name

        print(f"Loading {story_name} for subject {subject}")
        self.story_name = story_name
        self.subject = subject
        self.sub_nc_mask = sub_nc_mask
        self.read_stim_dir = read_stim_dir
        self.read_fmri_dir = read_fmri_dir
        self.delays = delays
        self.wav_params = wav_params
        self.wav_tensor = torch.tensor(np.load(os.path.join(self.read_stim_dir, f"{story_name}/wav.npy")))
        self.wav_feat = extract_speech_times(self.wav_tensor, self.wav_params['chunksz_sec'], self.wav_params['contextsz_sec'],
                                             sampling_rate=self.wav_params['sampling_rate'])
        
        self.trim_start = trim_start; self.trim_end = trim_end
        

    
    def fetch_data(self):
        # print(f'processing {self.story_name} for subject {self.subject}')
        self.aligned_wav = get_inds2tr_map(self.wav_tensor, self.wav_feat['indices'], self.wav_feat['times'],
                                      wordseqs[self.story_name].tr_times + 1, used_chuncksz_sec=self.wav_params['chunksz_sec'])
        
        self.delayed_wav = make_delayed(self.aligned_wav, self.delays).float()
        self.fmri_tensor = torch.tensor(self._load_h5py(os.path.join(self.read_fmri_dir,
                                                                     f'UTS0{self.subject}', f"{self.story_name}.hf5"))).float()
        
        assert self.delayed_wav.shape[0] == self.fmri_tensor.shape[0], "Wav and FMRI tensor should have the same time dimension"
    def __getitem__(self, index):

        return self.delayed_wav[index], self.fmri_tensor[index]
  
    def __len__(self):
        return len(wordseqs[self.story_name].tr_times[self.trim_start:-self.trim_end])#self.fmri_tensor.shape[0]
    
    def _load_h5py(self, file_path, key=None):   

        data = dict()
        with h5py.File(file_path) as hf:
            if key is None:
                for k in hf.keys():
                    print("{} will be loaded".format(k))
                    data[k] = list(hf[k])
            else:
                data[key] = hf[key]
        return np.nan_to_num(np.array(data['data']))[:, self.sub_nc_mask] # mask out the nans and choose the most sig voxels

    def _clear(self):
        self.aligned_wav = None
        self.delayed_wav = None
        self.fmri_tensor = None
  

class FMRIRandomStory(Dataset):
    def __init__(self,
                 story_name,
                 subject,
                 sub_nc_mask,
                 read_stim_dir='../datasets/processed_stim_data_dp',
                 read_fmri_dir='../../ds003020/derivative/preprocessed_data/', 
                 delays=range(1, ndelays + 1),
                 trim_start=10,
                 trim_end=5,
                 random_method='all',
                 random_chunklen=15,
                 random_wind=10,
                    
                 **wav_params):
        super(FMRIRandomStory, self).__init__()
        self.story_name = story_name

        print(f"Loading {story_name} for subject {subject}")
        self.story_name = story_name
        self.subject = subject
        self.sub_nc_mask = sub_nc_mask
        self.read_stim_dir = read_stim_dir
        self.read_fmri_dir = read_fmri_dir
        self.delays = delays
        self.wav_params = wav_params
        self.wav_tensor = torch.tensor(np.load(os.path.join(self.read_stim_dir, f"{story_name}/wav.npy")))
        self.wav_feat = extract_speech_times(self.wav_tensor, self.wav_params['chunksz_sec'], self.wav_params['contextsz_sec'],
                                             sampling_rate=self.wav_params['sampling_rate'])
        
        self.trim_start = trim_start; self.trim_end = trim_end
        self.random_method = random_method
        self.random_chunklen = random_chunklen
        self.random_wind = random_wind
        
             

    def random_all(self, tensor1, tensor2):
        """
        Adjusts tensor2 to match the first dimension of tensor1.
        If tensor2's first dimension is less, randomly completes it by picking random indices from tensor2.
        If tensor2's first dimension is greater, trims it.
        
        Parameters:
        tensor1 (torch.Tensor): The reference tensor
        tensor2 (torch.Tensor): The tensor to adjust
        
        Returns:
        torch.Tensor: The adjusted tensor2
        """
        dim1_size = tensor1.size(0)
        dim2_size = tensor2.size(0)
        
        if dim2_size < dim1_size:
            # Randomly select indices to complete tensor2
            random_indices = torch.randint(0, dim2_size, (dim1_size - dim2_size,))
            additional_rows = tensor2[random_indices]
            adjusted_tensor2 = torch.cat((tensor2, additional_rows), dim=0)
        elif dim2_size > dim1_size:
            # Trim tensor2 to match the first dimension of tensor1
            adjusted_tensor2 = tensor2[:dim1_size]
        else:
            adjusted_tensor2 = tensor2
        
        return adjusted_tensor2

    def random_local(self, tensor1, tensor2):
        """
        Shuffles the first dimension of tensor2.
        
        Parameters:
        tensor1 (torch.Tensor): The reference tensor
        tensor2 (torch.Tensor): The tensor to shuffle
        
        Returns:
        torch.Tensor: The shuffled tensor2
        """
        dim1_size = tensor1.size(0)
        assert tensor2.size(0) == dim1_size, "The first dimensions of tensor1 and tensor2 must match."
        
        shuffled_indices = torch.randperm(dim1_size)
        shuffled_tensor2 = tensor2[shuffled_indices]
        
        return shuffled_tensor2

    def shuffle_chunks_efficient(self, input_tensor, tensor2):
        """
        Efficiently shuffle the rows of the input array in chunks.

        Args:
        - input_tensor (torch.Tensor): The input array to shuffle.
        - chunklen (int): The length of each chunk.
        - wind (int): The minimum distance in chunks between swapped chunks.

        Returns:
        - np.ndarray: The shuffled array.
        """
        
        chunklen = self.random_chunklen
        wind = self.random_wind
        original_len = len(input_tensor)
        num_chunks = original_len // chunklen  # Number of complete chunks
        truncated_len = num_chunks * chunklen 
        input_tensor = input_tensor[:truncated_len]
        tensor2 = tensor2[:truncated_len]
        
        arr = np.arange(len(input_tensor))
        num_chunks = len(input_tensor) // chunklen
        
        if len(arr) % chunklen != 0:
            raise ValueError("The length of the array must be divisible by chunklen.")
        
        # Split the array into chunks
        chunks = [arr[i*chunklen:(i+1)*chunklen] for i in range(num_chunks)]
        
        # Generate a list of indices
        indices = list(range(num_chunks))
        
        # Modified Fisher-Yates shuffle to respect the wind constraint
        for i in range(num_chunks):
            valid_indices = [j for j in range(i, num_chunks) if j >= i + wind or j <= i - wind]
            if not valid_indices:
                print('------- could not shuffle, not randomized ------')
                continue
            swap_idx = np.random.choice(valid_indices)
            indices[i], indices[swap_idx] = indices[swap_idx], indices[i]
        
        # Reassemble the array with the shuffled chunks
        shuffled_arr = np.concatenate([chunks[i] for i in indices])
        
        return input_tensor[shuffled_arr], tensor2

    def fetch_data(self):
        # print(f'processing {self.story_name} for subject {self.subject}')
        self.aligned_wav = get_inds2tr_map(self.wav_tensor, self.wav_feat['indices'], self.wav_feat['times'],
                                      wordseqs[self.story_name].tr_times, used_chuncksz_sec=self.wav_params['chunksz_sec'])
        
        self.delayed_wav = make_delayed(self.aligned_wav, self.delays).float()
        if self.random_method == 'all':
            story_name = np.random.choice(train_stories)
        elif self.random_method == 'local':
            story_name = self.story_name
        else:
            raise ValueError("Invalid random method. Choose either 'all' or 'local'")
            
        self.fmri_tensor = torch.tensor(self._load_h5py(os.path.join(self.read_fmri_dir,
                                                                     f'UTS0{self.subject}', f"{story_name}.hf5"))).float()
        if self.random_method == 'all':
            self.fmri_tensor = self.random_all(self.delayed_wav, self.fmri_tensor)
        elif self.random_method == 'local':
            self.fmri_tensor, self.delayed_wav = self.shuffle_chunks_efficient(self.fmri_tensor, self.delayed_wav)
            
        assert self.delayed_wav.shape[0] == self.fmri_tensor.shape[0], "Wav and FMRI tensor should have the same time dimension"
        # print(f'wav tensor shape: {self.delayed_wav.shape}, fmri tensor shape: {self.fmri_tensor.shape}')
    def __getitem__(self, index):
        ## return wav, TR
        ## select from the wav tensor and TR tensor
        return self.delayed_wav[index], self.fmri_tensor[index]
  
    def __len__(self):
        original_len = (len(wordseqs[self.story_name].tr_times))
        num_chunks = original_len // self.random_chunklen  # Number of complete chunks
        truncated_len = num_chunks * self.random_chunklen 
        return (len(wordseqs[self.story_name].tr_times[:truncated_len][self.trim_start:-self.trim_end]))
    
    def _load_h5py(self, file_path, key=None):   

        data = dict()
        with h5py.File(file_path) as hf:
            if key is None:
                for k in hf.keys():
                    print("{} will be loaded".format(k))
                    data[k] = list(hf[k])
            else:
                data[key] = hf[key]
        return np.nan_to_num(np.array(data['data']))[:, self.sub_nc_mask] # mask out the nans and choose the most sig voxels

    def _clear(self):
        self.aligned_wav = None
        self.delayed_wav = None
        self.fmri_tensor = None


class WhisperStory(Dataset):
    def __init__(self,
                 story_name,        
                 read_stim_dir='../datasets/processed_stim_data_dp',
                 wsave_dir='../datasets/processed_stim_whisper_embeds',
                 delays=range(1, ndelays + 1),
                 trim_start=10,
                 trim_end=5,
                 **wav_params):
        super(WhisperStory, self).__init__()
        self.story_name = story_name
        # 
        print(f"Loading {story_name}")
        self.story_name = story_name
        self.read_stim_dir = read_stim_dir
        self.wsave_dir = wsave_dir
        self.delays = delays
        self.wav_params = wav_params
        self.wav_tensor = torch.tensor(np.load(os.path.join(self.read_stim_dir, f"{story_name}/wav.npy")))
        self.wav_feat = extract_speech_times(self.wav_tensor, self.wav_params['chunksz_sec'], self.wav_params['contextsz_sec'],
                                             sampling_rate=self.wav_params['sampling_rate'])
        
        
        self.trim_start = trim_start; self.trim_end = trim_end

    
    def fetch_data(self):
        # print(f'processing {self.story_name} for subject {self.subject}')
        self.aligned_wav = get_inds2tr_map(self.wav_tensor, self.wav_feat['indices'], self.wav_feat['times'],
                                      wordseqs[self.story_name].tr_times, used_chuncksz_sec=self.wav_params['chunksz_sec'])
        
        self.delayed_wav = make_delayed(self.aligned_wav, self.delays).float()
        self.embed_tensor = torch.tensor(np.load(os.path.join(self.wsave_dir, f"{self.story_name}.npy")))
        
        assert self.delayed_wav.shape[0] == self.embed_tensor.shape[0], "Wav and Embed tensor should have the same time dimension"
        


    def __getitem__(self, index):
        return self.delayed_wav[index], self.embed_tensor[index]
    
    def __len__(self):
        return len( self.delayed_wav) #len(wordseqs[self.story_name].tr_times[self.trim_start:-self.trim_end])#self.fmri_tensor.shape[0]
        
        
    
    def _clear(self):
        self.aligned_wav = None
        self.delayed_wav = None
        self.embed_tensor = None
