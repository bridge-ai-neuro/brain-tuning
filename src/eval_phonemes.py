from datasets import  Audio, Dataset as HFDataset

import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import torchaudio
from data_utils import *
import json
import numpy as np
import random
import pickle

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import WhisperTokenizer, WhisperProcessor, AutoProcessor
from transformers import WhisperFeatureExtractor
from transformers import HubertModel, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from transformers import WhisperModel, Wav2Vec2Model
from tokenizers.processors import TemplateProcessing

from whisper_linear import WhisperLinear
from hubert_linear import HubertLinear
from wav2vec_linear import Wav2VecLinear, Wav2VecLoRA


os.environ["WANDB_DISABLED"] = "true"

parser = ArgumentParser()
## model args
parser.add_argument('--model_name', type=str, default='openai/whisper-small', help='Name of the pre-trained model')
parser.add_argument('--nc_thr', type=float, default=0.4, help='Learning rate for the linear layer')
parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for training')
parser.add_argument('--num_epochs', type=int, default=50)
## data arg
parser.add_argument('--sampling_rate', type=int, default=16000, help='Sampling rate for the wav file')



parser.add_argument('--model_path', type=str, default='../outputs/', help='saved model path')
parser.add_argument('--subject', type=str, default='3', help='Subject number to train on')
parser.add_argument('--data_path', type=str, default='../datasets/data_phonem/',)
parser.add_argument('--is_whisper', action='store_true',)
parser.add_argument('--is_hubert', action='store_true',)
parser.add_argument('--is_wembed', action='store_true',)
parser.add_argument('--is_lora', action='store_true', help='Use LoRA model')



## logs args
parser.add_argument('--save_dir', type=str, default='../outputs/phonems_preds')
parser.add_argument('--model_suf', type=str, default='eval_phon_wav2vec', help='Model suffix')
parser.add_argument('--exp_name', type=str, default='eval_phon_wav2vec', help='Experiment name')

args = parser.parse_args()

timit_path = args.data_path
data_path = args.data_path
device = args.device

layers = [2, 5, 7, 8, 10, 11, 12]
exp_name='phonem'

from transformers import Wav2Vec2ForPreTraining

def load_ssl_model(model_path):
    model = Wav2Vec2ForPreTraining.from_pretrained(model_path)
    model.eval()
    return model.wav2vec2

def get_lora_keys(state_dict):
    lora_state_dict = {}
    for key in state_dict.keys():
        if 'lora_model' in key:
            lkey = key.replace('module.lora_model.', '')
            lora_state_dict[lkey] = state_dict[key]
    return lora_state_dict


def init_base_models(model_name): #TODO move to utils and move data processors here
    #TODO: also save a processed version of the dataset
    
    if 'hubert' in model_name:
        lm = HubertModel.from_pretrained(model_name)
        lm.feature_extractor._freeze_parameters()
    elif 'whisper' in model_name:
        lm = WhisperModel.from_pretrained(model_name)
    elif 'wav2vec' in model_name:
        print('loading wav2vec model: {}'.format(model_name))
        lm = Wav2Vec2Model.from_pretrained(f'facebook/{model_name}')
    else:
        raise ValueError('Model type not found')
    
    return lm

# %%
df_train = pd.read_csv(os.path.join(timit_path, 'train_data.csv'))
df_test = pd.read_csv(os.path.join(timit_path, 'test_data.csv'))
df = pd.concat([df_train, df_test])
df = df[df['is_converted_audio'] == False]


train = json.load(open(os.path.join(timit_path, 'custom_train.json')))
valid = json.load(open(os.path.join(timit_path, 'custom_valid.json')))
test = json.load(open(os.path.join(timit_path, 'custom_test.json')))

print(f"Duration of Train: {get_durations(train) // 60} mns")
print(f"Duration of Valid: {get_durations(valid) // 60} mns")
print(f"Duration of Test : {get_durations(test) // 60} mns")

train = convert_to_feature_dict(train)
valid = convert_to_feature_dict(valid)
test  = convert_to_feature_dict(test)

# %%
train_dataset = HFDataset.from_dict(train)
valid_dataset = HFDataset.from_dict(valid)
test_dataset = HFDataset.from_dict(test)

# %%
print(train_dataset)

train_dataset = (train_dataset
                 .map(prepare_text_data)
                 .remove_columns(["word_file", "phonetic_file"]))
valid_dataset = (valid_dataset
                 .map(prepare_text_data)
                 .remove_columns(["word_file", "phonetic_file"]))
test_dataset  = (test_dataset
                 .map(prepare_text_data)
                 .remove_columns(["word_file", "phonetic_file"]))

# %%
train_phonetics = [phone for x in train_dataset for phone in x['phonetic'].split()]
print("num of train phones:\t", len(set(train_phonetics)))

# %%
# TimitBet 61 phoneme mapping to 39 phonemes
# by Lee, K.-F., & Hon, H.-W. (1989). Speaker-independent phone recognition using hidden Markov models. IEEE Transactions on Acoustics, Speech, and Signal Processing, 37(11), 1641–1648. doi:10.1109/29.46546 
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

# %%
train_dataset = train_dataset.map(normalize_phones)
valid_dataset = valid_dataset.map(normalize_phones)
test_dataset = test_dataset.map(normalize_phones)

# %%
train_phonetics = [phone for x in train_dataset for phone in x['phonetic'].split()]
valid_phonetics = [phone for x in valid_dataset for phone in x['phonetic'].split()]
test_phonetics = [phone for x in test_dataset for phone in x['phonetic'].split()]

print("num of train phones:\t", len(set(train_phonetics)))
print("num of valid phones:\t", len(set(valid_phonetics)))
print("num of test phones:\t", len(set(test_phonetics)))

# %%
phone_vocabs = set(train_phonetics)
phone_vocabs.remove('h#')
phone_vocabs = sorted(phone_vocabs)

def count_frequency(phonetics):
    phone_counts = {phone: 0 for phone in phone_vocabs}
    for phone in phonetics:
        if phone in phone_vocabs:
            phone_counts[phone] += 1
    # eliminate h# for visualization purposes
    return [phone_counts[phone] for phone in phone_vocabs] 

# %%
train_phone_counts = count_frequency(train_phonetics)
valid_phone_counts = count_frequency(valid_phonetics)
test_phone_counts  = count_frequency(test_phonetics)

# %%
train_phone_ratio = [count / sum(train_phone_counts) for count in train_phone_counts]
valid_phone_ratio = [count / sum(valid_phone_counts) for count in valid_phone_counts]
test_phone_ratio  = [count / sum(test_phone_counts) for count in test_phone_counts]

# %% [markdown]
# ## Load Audio File

# %%
train_dataset = (train_dataset
                 .cast_column("audio_file", Audio(sampling_rate=16_000))
                 .rename_column('audio_file', 'audio'))
valid_dataset = (valid_dataset
                 .cast_column("audio_file", Audio(sampling_rate=16_000))
                 .rename_column('audio_file', 'audio'))
test_dataset = (test_dataset
                 .cast_column("audio_file", Audio(sampling_rate=16_000))
                 .rename_column('audio_file', 'audio'))

rand_int = random.randint(0, len(train_dataset)-1)

print("Text:", train_dataset[rand_int]["text"])
print("Phonetics:", train_dataset[rand_int]["phonetic"])
print("Input array shape:", train_dataset[rand_int]["audio"]["array"].shape)
print("Sampling rate:", train_dataset[rand_int]["audio"]["sampling_rate"])
# ipd.Audio(data=train_dataset[rand_int]["audio"]["array"], autoplay=False, rate=16000)

vocab_train = list(set(train_phonetics)) + [' ']
vocab_valid = list(set(valid_phonetics)) + [' ']
vocab_test  = list(set(test_phonetics)) + [' ']

vocab_dict = json.load(open(f'{data_path}/vocab.json'))


# %%
symbols = {"a": "ə", "ey": "eɪ", "aa": "ɑ", "ae": "æ", "ah": "ə", "ao": "ɔ",
           "aw": "aʊ", "ay": "aɪ", "ch": "ʧ", "dh": "ð", "eh": "ɛ", "er": "ər",
           "hh": "h", "ih": "ɪ", "jh": "ʤ", "ng": "ŋ",  "ow": "oʊ", "oy": "ɔɪ",
           "sh": "ʃ", "th": "θ", "uh": "ʊ", "uw": "u", "zh": "ʒ", "iy": "i", "y": "j"}


symbols2idx = {}

if args.is_hubert:

    model_name = "facebook/hubert-base-ls960"
    
elif args.is_whisper:
    model_name = 'openai/whisper-small'
else:
    model_name = 'facebook/wav2vec2-base'
    
    

if args.is_hubert:
    processor =  Wav2Vec2FeatureExtractor.from_pretrained(model_name, return_tensors="pt")#(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)  
elif args.is_whisper:
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"{model_name}")
    tokenizer = WhisperTokenizer.from_pretrained(f"{model_name}", language="English", task="transcribe", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

else:
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(f"{data_path}/", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", )  # './' load vocab.json in the current directory
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)  
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def prepare_dataset(batch):
    audio = batch["audio"]
    
    # batched output is "un-batched"
    if args.is_whisper:
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    else:
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        
    batch["input_length"] = len(batch["input_values"])
    
    # with processor.as_target_processor():
    # print(batch["phonetic"])
    batch["labels"] = np.array([vocab_dict[k] for k in (batch["phonetic"].split(' '))])
    
    return batch

train_dataset = train_dataset.map(prepare_dataset)
valid_dataset = valid_dataset.map(prepare_dataset)
test_dataset = test_dataset.map(prepare_dataset)


best_mean = args.model_path
model_suf = args.model_suf
save_root = f'{args.data_path}/phonem/'

if 'base' in (args.subject):
    if 'ssl' in args.subject:
        print('loading SSL model from ckpt {}'.format(best_mean))
        lm = load_ssl_model(args.model_path)
    else:
        lm = init_base_models(model_name)
elif args.is_lora:
    nc = get_n_voxels(int(args.subject))
    nv = len(nc[nc>args.nc_thr])
    ckpt = torch.load(best_mean)
    ckpt_w = get_lora_keys(ckpt)
    print(f'loading lora model with {best_mean}')
    lm = Wav2VecLoRA(out_dim=nv,
                     wav2vec_model_name=model_name
                     ).lora_model
    lm.load_state_dict(ckpt_w)        
else:
    ckpt = torch.load(best_mean)
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
        lm = WhisperLinear(nv)
    else:
        # d_roi = {'2': 4889, '1': 2130, '3': 2542}
        lm = Wav2VecLinear(nv, wav2vec_model_name=model_name).wav2vec
        # lm = Wav2VecLinear(d_roi[args.subject], wav2vec_model_name=model_name).wav2vec
    
        for k in ckpt:
            if k not in ['module.linear.weight', 'module.linear.bias']:
                ckpt_w[k.replace('module.wav2vec.', '')] = ckpt[k]


                    
# if 'module.' in list(ckpt.keys())[0]:
#     ## for data parallel model
#     ckpt = {key.replace('module.', ''): value for key, value in ckpt.items()}
if 'base' not in args.subject:

    print(f'loading model with {best_mean}')
    lm.load_state_dict(ckpt_w)
lm.eval()
lm = lm.to(device)

if args.is_hubert:
    
    features_dict_tr, labels_tr = get_probing_data(train_dataset, lm, layers, device)
    save_pho_data(features_dict_tr, labels_tr, layers, phase='train', model_suf=model_suf, save_root=save_root)

    features_dict, labels = get_probing_data(test_dataset, lm, layers, device)
    save_pho_data(features_dict, labels, layers, phase='test', model_suf=model_suf, save_root=save_root)

elif args.is_whisper:
    features_dict_tr, labels_tr = get_probing_data_whisper(train_dataset, lm, layers, device)
    save_pho_data(features_dict_tr, labels_tr, layers, phase='train', model_suf=model_suf, save_root=save_root)

    features_dict, labels = get_probing_data_whisper(test_dataset, lm, layers, device) # TODO why is it different from hubert
    save_pho_data(features_dict, labels, layers, phase='test', model_suf=model_suf, save_root=save_root)     

else:
    
    features_dict_tr, labels_tr = get_probing_data(train_dataset, lm, layers, device)
    save_pho_data(features_dict_tr, labels_tr, layers, phase='train', model_suf=model_suf, save_root=save_root)

    features_dict, labels = get_probing_data(test_dataset, lm, layers, device)
    save_pho_data(features_dict, labels, layers, phase='test', model_suf=model_suf, save_root=save_root)

       

per_ly_preds = {} 
for layer in layers:
    phase = 'train'
    Xtr =  np.load(f"{save_root}/{exp_name}_{model_suf}_{phase}_layer{layer}_features.npy")
    ytr = np.load(f"{save_root}/{model_suf}_{phase}_labels.npy",)

    phase = 'test'
    Xt =  np.load(f"{save_root}/{exp_name}_{model_suf}_{phase}_layer{layer}_features.npy")
    yt = np.load(f"{save_root}/{model_suf}_{phase}_labels.npy",)
    print(f'layer {layer} process')
    p = train_layer_model(Xtr, ytr, Xt, yt, num_epochs=args.num_epochs, num_features=Xtr.shape[1], device=device)
    per_ly_preds[layer] = p


## save preds
if args.is_wembed:
    save_dst = f"{args.save_dir}/LLama2_7b/"
    os.makedirs(save_dst, exist_ok=True)    
    with open(f"{args.save_dir}/LLama2_7b/{exp_name}_{model_suf}_preds.pkl", 'wb') as f:
        pickle.dump(per_ly_preds, f)
        
else:  
    save_dst = f"{args.save_dir}/UTS0{args.subject}"
    os.makedirs(save_dst, exist_ok=True)    
    with open(f"{args.save_dir}/UTS0{args.subject}/{exp_name}_{model_suf}_preds.pkl", 'wb') as f:
        pickle.dump(per_ly_preds, f)