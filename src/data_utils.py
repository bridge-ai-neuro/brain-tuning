
import cortex 
import numpy as np 
import librosa
import numpy as np
import pandas as pd 
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import json
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
from torch.utils.data import DataLoader, Dataset
import torchaudio
import glob
from torch.nn.utils.rnn import pad_sequence
import json
import torch
from hubert_linear import HubertLinear 
from whisper_linear import WhisperLinear 
from wav2vec_linear import Wav2VecLinear
from transformers import HubertModel, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from transformers import WhisperModel, Wav2Vec2Model


def init_base_models(model_name):
    
    if 'hubert' in model_name:
        lm = HubertModel.from_pretrained(model_name)
        lm.feature_extractor._freeze_parameters()
    elif 'whisper' in model_name:
        lm = WhisperModel.from_pretrained(model_name)
    elif 'wav2vec' in model_name:
        lm = Wav2Vec2Model.from_pretrained(f'facebook/{model_name}')
    else:
        raise ValueError('Model type not found')
    
    return lm

def load_tuned_model(model_name, best_mean, args):
    ckpt = (torch.load(best_mean))
    
    ckpt_w = {}

    nc = get_n_voxels(int(args.subject))

    if args.is_wembed:
        nv = 25600
    else:
        nv = len(nc[nc>args.nc_thr])
        
    if args.is_hubert:
        lm = HubertLinear(nv, model_name=model_name).hubert
        print(f'loading hubert model with {best_mean}')
        for k in ckpt:
            if k not in ['module.linear.weight', 'module.linear.bias']:
                ckpt_w[k.replace('module.hubert.', '')] = ckpt[k]


    elif args.is_whisper:
        lm = WhisperLinear(nv).whisper_encoder
        print(f'loading whisper model with {best_mean}')
        for k in ckpt:
            if k not in ['module.linear.weight', 'module.linear.bias']:
                ckpt_w[k.replace('module.whisper_encoder.', '')] = ckpt[k]
        
    else:
        lm = Wav2VecLinear(nv, wav2vec_model_name=model_name).wav2vec
        for k in ckpt:
            if k not in ['module.linear.weight', 'module.linear.bias']:
                ckpt_w[k.replace('module.wav2vec.', '')] = ckpt[k]
    
    return lm, ckpt_w
    
    

class SimpleLinearModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.linear(x)

class TwoLinearModel(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64):
        super(TwoLinearModel, self).__init__()
        self.linear1 = nn.Linear(num_features, hidden_dim)
        self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.relu2 = nn.ReLU()
        
        self.linear = nn.Linear(hidden_dim, num_classes)
        
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.linear2(x)
        # x = self.relu2(x)
        return self.linear(x)    
    
def collate_fn(batch):
    # Separate the audio sequences and labels
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Pad sequences
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    # Convert labels to a tensor (if they are not already)
    labels_tensor = torch.tensor(labels)

    return sequences_padded, labels_tensor
  

class EmotionsDataset(Dataset):
    def __init__(self, processor, proc_key, split='training', data_path='datasets/AudioWAV'):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.split = split
        self.map_dict = {'ANG':0, 'DIS':1, 'FEA':2, 'HAP':3, 'NEU':4, 'SAD':5}
        self.processor = processor
        self.proc_key = proc_key
        self.data_files = list(glob.glob(f'{data_path}/{split}/*.wav'))
        self.n_classes = len(self.map_dict)
    # torchaudio.datasets.SPEECHCOMMANDS('datasets/', download=False, subset='testing',)
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        audio_file = self.data_files[idx]
        waveform, sample_rate = torchaudio.load(audio_file)
        label_str = audio_file.split('/')[-1].split('_')[-2]
        label = self.map_dict[label_str]
        if self.proc_key == 'input_features':
            input_vals = self.processor(waveform[0].numpy(), return_tensors="pt", sampling_rate=sample_rate)[self.proc_key][0]
                        
        else:
            input_vals = self.processor(waveform[0].numpy(), return_tensors="pt", sampling_rate=sample_rate, max_length=32000, padding="max_length", truncation=True)[self.proc_key][0]
        return input_vals, label
    
    
class CommandDataset(Dataset):
    def __init__(self, processor, proc_key, split='training', data_path='../datasets/', map_dict=None):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.split = split
        self.map_dict = map_dict
        self.processor = processor
        self.proc_key = proc_key
        self.data = torchaudio.datasets.SPEECHCOMMANDS(data_path, subset=split)
    # torchaudio.datasets.SPEECHCOMMANDS('datasets/', download=False, subset='testing',)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_number = self.data[idx]

        input_vals = self.processor(waveform[0], return_tensors="pt", sampling_rate=sample_rate)[self.proc_key][0]
        if self.map_dict:
            label = self.map_dict[label]
        return input_vals, label

import os
import json
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

class SLURPDataset(Dataset):
    def __init__(self, json_file, audio_dir, processor, proc_key):
        jsonObj = pd.read_json(path_or_buf=json_file, lines=True)
        self.data = json.loads(jsonObj.to_json(orient="records"))
        # self.data = self.parse_json(json_file)
        self.audio_dir = audio_dir
        self.processor = processor
        self.sample_rate = 16000
        self.proc_key = proc_key
        self.label_to_idx = {label: idx for idx, label in enumerate(set(item['action'] for item in self.data))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Using the first audio recording for simplicity
        audio_path = os.path.join(self.audio_dir, item['recordings'][0]['file'])
        # audio_path = os.path.join(self.audio_dir, item['file'])
        
        audio, sample_rate = torchaudio.load(audio_path)
        label = item['action']
        label = self.label_to_idx[label]
        
        if self.proc_key == 'input_features':

            audio = self.processor(
                                audio[0],
                                return_tensors="pt",
                                sampling_rate=sample_rate
                                )[self.proc_key][0]
        else:
            audio = self.processor(
                                audio[0], return_tensors="pt",
                                sampling_rate=sample_rate,
                                max_length=int(7 * sample_rate),
                                padding="max_length",
                                truncation=True
                                    
                                )[self.proc_key][0]
            
        
        return audio, label

    def parse_json(self, json_file):
        jsonObj = pd.read_json(path_or_buf=json_file, lines=True)
        data = json.loads(jsonObj.to_json(orient="records"))
        parsed_data = []
        for d in data:
            a = d['action']
            i = d['intent']
            for r in d['recordings']:
                parsed_data.append({'action': a, 'intent': i, 'file': r['file']})
        return parsed_data


def get_slurp_loader(processor, proc_key, batch_size=64, num_workers=8):
    train_json = '../datasets/slurp/dataset/slurp/train.jsonl'
    test_json = '../datasets/slurp/dataset/slurp/test.jsonl'
    audio_dir = '../datasets/slurp/scripts/audio/slurp_real'
    train_dataset = SLURPDataset(json_file=train_json, audio_dir=audio_dir, processor=processor, proc_key=proc_key)
    test_dataset = SLURPDataset(json_file=test_json, audio_dir=audio_dir, processor=processor, proc_key=proc_key)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
    return train_loader, test_loader    
    
def get_commands_loader(processor, proc_key, data_path='../datasets/', batch_size=64, num_workers=8):
    mp_path = '../datasets/label2idx.json'
    map_dict = json.load(open(mp_path, 'r'))
    train_dataset = CommandDataset(processor, proc_key, split='training', data_path=data_path, map_dict=map_dict)
    test_dataset = CommandDataset(processor, proc_key, split='testing', data_path=data_path, map_dict=map_dict)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
    return train_loader, test_loader    
    
def get_emotions_loader(processor, proc_key, data_path='../datasets/AudioWAV', batch_size=64, num_workers=8):

    train_dataset = EmotionsDataset(processor, proc_key, split='training', data_path=data_path)
    test_dataset = EmotionsDataset(processor, proc_key, split='testing', data_path=data_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,)# collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,)# collate_fn=collate_fn)
    
    return train_loader, test_loader    
    
    
def get_voxels_nc_mask(subject, thr=0.4, data_path='../datasets'):
    sub_nc = np.load(f'{data_path}/subject_NCs/UTS0{subject}.npy')
    sub_neur_mask = np.where(sub_nc > thr)[0]
    out_dim = len(sub_neur_mask)  # Desired output dimension
    return sub_neur_mask, out_dim

def get_n_voxels(subject):
    sub_nc = np.load(f'../datasets/subject_NCs/UTS0{subject}.npy')
    return sub_nc


def get_roi_fmri_mask(subject, roi=None, data_path='../datasets'):
    '''
    for the given ROI, get the indices of the voxels in the subject's brain predictions/fmri data
    '''
    sub_nc = np.load(f'{data_path}/subject_NCs/UTS0{subject}.npy')
    
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
    

def get_roi_inds(subject, ROIS = ['Broca', 'AC',], data_path='../datasets'): # 'RSC', 'S2M', 'PPA']):
    rois_inds = []
    for roi in ROIS:
        rinds = get_roi_fmri_mask(subject, roi, data_path=data_path)
        rois_inds.extend(rinds)
    
    return rois_inds

def filter_rois_data(data, rois_inds):
    if len(data.shape) == 1:
        return data[rois_inds]
    else:
        return data[:, rois_inds]
    


def get_durations(dict_data, save_dir='../datasets/data_phonem/'):
    total_durations = 0

    for entry in dict_data.values():
        audio_data, _ = librosa.load(entry['audio_file'].replace('data', save_dir), sr=16_000)
        duration = len(audio_data) / 16_000
        total_durations += duration

    return int(total_durations)



def convert_to_feature_dict(data_dict, save_dir='../datasets/data_phonem/'):
    # convert each feature into an array instead
    audio_files = []
    word_files = []
    phonetic_files = []
    for key, value in data_dict.items():
        audio_files.append(value['audio_file'].replace('data', save_dir))
        word_files.append(value['word_file'].replace('data', save_dir))
        phonetic_files.append(value['phonetic_file'].replace('data', save_dir))
    
    return {
        'audio_file': audio_files,
        'word_file': word_files,
        'phonetic_file': phonetic_files
    }


def read_text_file(filepath):
    with open(filepath) as f:
        tokens = [line.split()[-1] for line in f]
        return " ".join(tokens)
    
def prepare_text_data(item):
    item['text'] = read_text_file(item['word_file'])
    item['phonetic'] = read_text_file(item['phonetic_file'])
    return item

phon61_map39 = {
    'iy':'iy',  'ih':'ih',   'eh':'eh',  'ae':'ae',    'ix':'ih',  'ax':'ah',   'ah':'ah',  'uw':'uw',
    'ux':'uw',  'uh':'uh',   'ao':'aa',  'aa':'aa',    'ey':'ey',  'ay':'ay',   'oy':'oy',  'aw':'aw',
    'ow':'ow',  'l':'l',     'el':'l',  'r':'r',      'y':'y',    'w':'w',     'er':'er',  'axr':'er',
    'm':'m',    'em':'m',     'n':'n',    'nx':'n',     'en':'n',  'ng':'ng',   'eng':'ng', 'ch':'ch',
    'jh':'jh',  'dh':'dh',   'b':'b',    'd':'d',      'dx':'dx',  'g':'g',     'p':'p',    't':'t',
    'k':'k',    'z':'z',     'zh':'sh',  'v':'v',      'f':'f',    'th':'th',   's':'s',    'sh':'sh',
    'hh':'hh',  'hv':'hh',   'pcl':'h#', 'tcl':'h#', 'kcl':'h#', 'qcl':'h#','bcl':'h#','dcl':'h#',
    'gcl':'h#','h#':'h#',  '#h':'h#',  'pau':'h#', 'epi': 'h#','nx':'n',   'ax-h':'ah','q':'h#' 
}


def convert_phon61_to_phon39(sentence):
    tokens = [phon61_map39[x] for x in sentence.split()]
    return " ".join(tokens)

def normalize_phones(item):
    item['phonetic'] = convert_phon61_to_phon39(item['phonetic'])
    return item


def save_num_pho_data(features_dict, labels, layers , phase='test', save_root=f'../datasets/data_phonem/phonem/', model_suf='base', exp_name='num_phonem'):
    for layer in layers:
        np.save(f"{save_root}/{exp_name}_{model_suf}_{phase}_layer{layer}_features.npy", features_dict[layer])
    
    labels_arr = np.zeros((len(labels), 40))
    for idx, lbl in enumerate(labels):
        labels_arr[idx, lbl] = 1
    
    labels_arr = labels_arr[:, 1:].astype(int)
    
    np.save(f"{save_root}/{model_suf}_{phase}_labels.npy", np.array(labels_arr))


def save_pho_data(features_dict, labels, layers , phase='test', save_root=f'../datasets/data_phonem/phonem/', model_suf='base', exp_name='phonem'):
    for layer in layers:
        np.save(f"{save_root}/{exp_name}_{model_suf}_{phase}_layer{layer}_features.npy", features_dict[layer])
    
    labels_arr = np.zeros((len(labels), 40))
    for idx, lbl in enumerate(labels):
        labels_arr[idx, lbl] = 1
    
    labels_arr = labels_arr[:, 1:].astype(int)
    
    np.save(f"{save_root}/{model_suf}_{phase}_labels.npy", np.array(labels_arr))

def save_pho_diff_data(features_dict, labels, layers , phase='test', save_root=f'../datasets/data_phonem/phonem/', model_suf='base', exp_name='phonem'):
    for layer in layers:
        np.save(f"{save_root}/{exp_name}_{model_suf}_{phase}_layer{layer}_features.npy", features_dict[layer])
    
  
    labels_arr = np.array(labels).astype(int)
    
    np.save(f"{save_root}/{model_suf}_{phase}_labels.npy", np.array(labels_arr))

def save_word_cls_data(features_dict, labels, layers, vocab_dict, subject=None , phase='test', save_root=f'../datasets/data_phonem/phonem/', model_suf='base', exp_name='wordid_cls'):
    for layer in layers:
        if subject:
            np.save(f"{save_root}/{exp_name}_{model_suf}_{subject}_{phase}_layer{layer}_features.npy", features_dict[layer])
        else:
            np.save(f"{save_root}/{exp_name}_{model_suf}_{phase}_layer{layer}_features.npy", features_dict[layer])
            
    
    labels_arr = np.zeros((len(labels), len(vocab_dict)))
    for idx, lbl in enumerate(labels):
        labels_arr[idx, lbl] = 1
    
    labels_arr = labels_arr[:, :].astype(int)
    
    np.save(f"{save_root}/{exp_name}_{model_suf}_{phase}_labels.npy", np.array(labels_arr))
    
def save_word_data(features_dict, labels, layers, subject=None, phase='test', save_root=f'../datasets/data_phonem/phonem/', model_suf='base', exp_name='wordid'):
    for layer in layers:
        if subject:
            np.save(f"{save_root}/{exp_name}_{model_suf}_{subject}_{phase}_layer{layer}_features.npy", features_dict[layer])
        else:
            np.save(f"{save_root}/{exp_name}_{model_suf}_{phase}_layer{layer}_features.npy", features_dict[layer])
    
    labels_arr = np.array(labels)
    
    np.save(f"{save_root}/{exp_name}_{model_suf}_{phase}_labels.npy", np.array(labels_arr))


def save_sent_data(features_dict, labels, layers, phase='test', save_root=f'../datasets/data_phonem/phonem/', model_suf='base', exp_name='senttype'):
    for layer in layers:
        np.save(f"{save_root}/{exp_name}_{model_suf}_{phase}_layer{layer}_features.npy", features_dict[layer])
    
    labels_arr = np.array(labels)
    
    np.save(f"{save_root}/{exp_name}_{model_suf}_{phase}_labels.npy", np.array(labels_arr))
    
from sklearn.metrics import f1_score

def evaluate_phon(X, y, model, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = torch.sigmoid(model(X.to(device)))
        predicted_classes = (predictions >= 0.5).float()  # Apply threshold
        # print(predicted_classes)
        accuracy = (predicted_classes == y.to(device)).float().mean()
        print(f'Accuracy: {accuracy.item()}')

        f1_micro = f1_score(y.cpu(), predicted_classes.cpu(), average='micro')
        f1_macro = f1_score(y.cpu(), predicted_classes.cpu(), average='macro')
    # print(y.cpu(), predicted_classes.cpu())
    return accuracy, f1_micro, f1_macro

def evaluate_sent(X, y, model, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = torch.argmax(model(X.to(device)), dim=1)
        predicted_classes = (predictions).float()  # Apply threshold
        # print(predicted_classes)
        accuracy = (predicted_classes == y.to(device)).float().mean()

        f1_micro = f1_score(y.cpu(), predicted_classes.cpu(), average='micro')
        f1_macro = f1_score(y.cpu(), predicted_classes.cpu(), average='macro')

    return dict(accuracy=accuracy.cpu().numpy(), f1_micro=f1_micro, f1_macro=f1_macro)


def evaluate_wordid_cls(X, y, model, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = torch.sigmoid(model(X.to(device)))
        predicted_classes = (predictions >= 0.5).float()  # Apply threshold
        # print(predicted_classes)
        accuracy = (predicted_classes == y.to(device)).float().mean()
        print(f'Accuracy: {accuracy.item()}')

        f1_micro = f1_score(y.cpu(), predicted_classes.cpu(), average='micro')
        f1_macro = f1_score(y.cpu(), predicted_classes.cpu(), average='macro')

    return dict(accuracy=accuracy.cpu().numpy(), f1_micro=f1_micro, f1_macro=f1_macro)

def evaluate_wordid(X, y, model, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = (model(X.to(device)))
        predicted_classes = (predictions).float()  # Apply threshold
        # print(predicted_classes)
        mse_val = (predicted_classes - y.to(device)).float().pow(2).mean().cpu().item()
        # print(f'Accuracy: {mse_val.item()}')

        ## get correlation 
        corr_list = []
        for i, j in zip(y.cpu().numpy(), predicted_classes.cpu().numpy()):
            if np.isnan(np.corrcoef(i, j)[0, 1]):
                print(i, j)
            corr_list.append(np.corrcoef(i, j)[0, 1])
            
        corr_val = np.mean(corr_list)

    return dict(mse_val=mse_val, corr_val=corr_val)



def train_layer_model(Xtr,
                      ytr,
                      Xt,
                      yt,
                      num_epochs,
                      device,
                      num_features=768,
                      num_classes=39):
    per = {}

    model = SimpleLinearModel(num_features, num_classes)
    criterion = nn.BCEWithLogitsLoss() # This loss combines a Sigmoid layer and the BCELoss in one single class
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Assuming X_train, y_train are your features and labels
    X_train = torch.tensor(Xtr, dtype=torch.float32)
    y_train = torch.tensor(ytr, dtype=torch.float32)

    X_test = torch.tensor(Xt, dtype=torch.float32)
    y_test = torch.tensor(yt, dtype=torch.float32)
    
    # If using a GPU
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    

    # num_epochs = 20  # Number of epochs
    
    for epoch in tqdm(range(num_epochs)):
        model.train()  # Set the model to training mode
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+3) % 1 == 0:
            t = evaluate_phon(X_test, y_test, model, device=device)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, eval: {t}')
            per[epoch] = t
    
    return per

def train_worid_cls_model(Xtr,
                      ytr,
                      Xt,
                      yt,
                      num_epochs,
                      vocab_dict,
                      device,
                      num_features=768):
    per = {}
    num_classes = len(vocab_dict)
    model = SimpleLinearModel(num_features, num_classes)
    criterion = nn.BCEWithLogitsLoss() # This loss combines a Sigmoid layer and the BCELoss in one single class
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Assuming X_train, y_train are your features and labels
    X_train = torch.tensor(Xtr, dtype=torch.float32)
    y_train = torch.tensor(ytr, dtype=torch.float32)

    X_test = torch.tensor(Xt, dtype=torch.float32)
    y_test = torch.tensor(yt, dtype=torch.float32)
    
    # If using a GPU
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    

    # num_epochs = 20  # Number of epochs

    for epoch in tqdm(range(num_epochs)):
        model.train()  # Set the model to training mode
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+3) % 1 == 0:
            t = evaluate_wordid_cls(X_test, y_test, model, device=device)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, eval: {t}')
            per[epoch] = t
    return per


def train_layer_mfcc_model(
                    Xtr,
                    ytr,
                    Xt,
                    yt,
                    num_epochs,
                    device,
                    num_features=768,
                    n_mfcc=40,
                    mfcc_dim=198*2 + 2):
    per = {}
    print(ytr.shape)
    y_train_arr = ytr.reshape(-1, n_mfcc, mfcc_dim)
    y_test_arr = yt.reshape(-1, n_mfcc, mfcc_dim)
    res_array = np.zeros((num_epochs, n_mfcc, len(y_test_arr) ))
    mse_array = np.zeros((num_epochs, n_mfcc ))
    for mfcc_id in tqdm(range(n_mfcc)):
        model = SimpleLinearModel(num_features, mfcc_dim)
        criterion = nn.MSELoss() # This loss combines a Sigmoid layer and the BCELoss in one single class
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Assuming X_train, y_train are your features and labels
        X_train = torch.tensor(Xtr, dtype=torch.float32)
        y_train = torch.tensor(y_train_arr[:, mfcc_id], dtype=torch.float32)

        X_test = torch.tensor(Xt, dtype=torch.float32)
        y_test = torch.tensor(y_test_arr[:, mfcc_id], dtype=torch.float32)
        
        # If using a GPU
        model = model.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        

        # num_epochs = 20  # Number of epochs
        for epoch in (range(num_epochs)):
            model.train()  # Set the model to training mode
            
            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            t = evaluate_wordid(X_test, y_test, model, device=device)
            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, eval: {t}')
            res_array[epoch, mfcc_id, :] = t['corr_val']
            mse_array[epoch, mfcc_id] = t['mse_val']
    
    epochs_per = res_array.mean(axis=1).mean(axis=-1)
    epochs_mse = mse_array.mean(axis=-1)
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}], MSE: {epochs_mse[epoch]:.4f}, eval: {epochs_per[epoch]}')
        per[epoch] = dict(mse_val=epochs_mse[epoch], corr_val=epochs_per[epoch])
    return per


def train_layer_wordid_model(Xtr,
                      ytr,
                      Xt,
                      yt,
                      num_epochs,
                      device,
                      num_features=768,
                      embedding_dim=100):
    per = {}

    model = SimpleLinearModel(num_features, embedding_dim)
    criterion = nn.MSELoss() # This loss combines a Sigmoid layer and the BCELoss in one single class
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Assuming X_train, y_train are your features and labels
    X_train = torch.tensor(Xtr, dtype=torch.float32)
    y_train = torch.tensor(ytr, dtype=torch.float32)

    X_test = torch.tensor(Xt, dtype=torch.float32)
    y_test = torch.tensor(yt, dtype=torch.float32)
    
    # If using a GPU
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    

    # num_epochs = 20  # Number of epochs

    for epoch in tqdm(range(num_epochs)):
        model.train()  # Set the model to training mode
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+3) % 1 == 0:
            t = evaluate_wordid(X_test, y_test, model, device=device)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, eval: {t}')
            per[epoch] = t
    return per


def train_layer_command_model(
                    Xtr,
                    ytr,
                    Xt,
                    yt,
                    num_epochs,
                    device,
                    num_features=768,
                    n_classes=35):
    per = {}

    model = SimpleLinearModel(num_features, n_classes)
    # model = TwoLinearModel(num_features, n_classes)
    
    criterion = nn.CrossEntropyLoss() # This loss combines a Sigmoid layer and the BCELoss in one single class
    optimizer = optim.Adam(model.parameters())

    # Assuming X_train, y_train are your features and labels
    X_train = torch.tensor(Xtr, dtype=torch.float32)
    y_train = torch.tensor(ytr)

    X_test = torch.tensor(Xt, dtype=torch.float32)
    y_test = torch.tensor(yt)
    
    # If using a GPU
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    print(f'number of classes: {n_classes}, max of classes: {y_train.max()}')
    # num_epochs = 20  # Number of epochs

    for epoch in tqdm(range(num_epochs)):
        model.train()  # Set the model to training mode
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+3) % 1 == 0:
            t = evaluate_sent(X_test, y_test, model, device=device)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, eval: {t}')
            per[epoch] = t
    return per


def train_layer_senttype_model(
                    Xtr,
                    ytr,
                    Xt,
                    yt,
                    num_epochs,
                    device,
                    num_features=768,
                    n_classes=3):
    per = {}

    model = SimpleLinearModel(num_features, n_classes)
    criterion = nn.CrossEntropyLoss() # This loss combines a Sigmoid layer and the BCELoss in one single class
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Assuming X_train, y_train are your features and labels
    X_train = torch.tensor(Xtr, dtype=torch.float32)
    y_train = torch.tensor(ytr)

    X_test = torch.tensor(Xt, dtype=torch.float32)
    y_test = torch.tensor(yt)
    
    # If using a GPU
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    

    # num_epochs = 20  # Number of epochs

    for epoch in tqdm(range(num_epochs)):
        model.train()  # Set the model to training mode
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+3) % 1 == 0:
            t = evaluate_sent(X_test, y_test, model, device=device)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, eval: {t}')
            per[epoch] = t
    return per

def get_batch_probing_data(dataloader, model, layers, device):
    features_dict = {}
    labels = []
    
    for batch in tqdm(dataloader):
        # if np.sum(batch['labels']) <= 0:
        #     print('no embed')
        #     continue
        # lbls = batch['labels']
        input_values, lbls = batch
        # print(input_values.shape, lbls.shape)
        with torch.no_grad():
        #             "input_ids": padded_input_ids,
        # "attention_mask": padded_attention_masks,
        
            # input_dict = dict(input_ids=batch['input_ids'].to(device),
            #                   attention_mask=batch['attention_mask'].to(device),)
            # feat = model(**(input_dict), output_hidden_states=True)
            
            feat = model((input_values).to(device=device), output_hidden_states=True)
        labels.extend(lbls.cpu().numpy())
        

        for layer_idx, layer_activations in enumerate(feat['hidden_states']):
        # Only save layers that the user wants (if specified)
            # if sel_layers:
            if layer_idx not in layers: continue
            if layer_idx not in features_dict:
                features_dict[layer_idx] = []
            
            features_dict[layer_idx].append(torch.mean(layer_activations, dim=1).detach().cpu().numpy())
    
    ## Concatenate the results
    for layer_idx in features_dict:
        features_dict[layer_idx] = np.concatenate(features_dict[layer_idx], axis=0)
        print(features_dict[layer_idx].shape, np.array(labels).shape)
    return features_dict, labels

# layers = [2, 7, 8, 10, 12]
def get_probing_data(dataloader, model, layers, device):
    features_dict = {}
    labels = []
    
    for batch in tqdm(dataloader):
        # if np.sum(batch['labels']) <= 0:
        #     print('no embed')
        #     continue
        with torch.no_grad():
            if isinstance(batch['input_values'], dict):
                input_dict = {k: torch.tensor(v,  device=device) for k, v in batch['input_values'].items()}

                feat = model(**(input_dict), output_hidden_states=True)
            else:
                feat = model(torch.tensor(batch['input_values'], device=device).unsqueeze(0), output_hidden_states=True)
        labels.append(batch['labels'])
        

        for layer_idx, layer_activations in enumerate(feat['hidden_states']):
        # Only save layers that the user wants (if specified)
            # if sel_layers:
            if layer_idx not in layers: continue
            if layer_idx not in features_dict:
                features_dict[layer_idx] = []
            
            features_dict[layer_idx].append(torch.mean(layer_activations, dim=1).detach().cpu().numpy())
    
    ## Concatenate the results
    for layer_idx in features_dict:
        features_dict[layer_idx] = np.concatenate(features_dict[layer_idx], axis=0)
    return features_dict, labels

def get_probing_data_whisper(dataloader, model, layers, device='cuda:0'):
    features_dict = {}
    labels = []
    
    for batch in tqdm(dataloader):
        if np.sum(batch['labels']) == 0:
            print('no embed')
            continue
        with torch.no_grad():
            feat = model.whisper_encoder(torch.tensor(batch['input_values'], device=device).unsqueeze(0), output_hidden_states=True)
        labels.append(batch['labels'])
        

        for layer_idx, layer_activations in enumerate(feat['hidden_states']):
        # Only save layers that the user wants (if specified)
            # if sel_layers:
            if layer_idx not in layers: continue
            if layer_idx not in features_dict:
                features_dict[layer_idx] = []
            
            features_dict[layer_idx].append(torch.mean(layer_activations, dim=1).detach().cpu().numpy())
    
    ## Concatenate the results
    for layer_idx in features_dict:
        features_dict[layer_idx] = np.concatenate(features_dict[layer_idx], axis=0)
    return features_dict, labels
      

def load_glove_embeddings(path='../datasets/glove.6B/glove.6B.100d.txt'):
    embeddings_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict


def preprocess_text(text):
    # Tokenizes and converts to lower case
    return word_tokenize(text.lower())


def get_text_embedding(tokens, embeddings_dict):
    embeddings = []
    for token in tokens:
        if token in embeddings_dict:
            embeddings.append(embeddings_dict[token])
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(100)  # Assuming 100-dimensional embeddings
    