from datasets import load_dataset
from tqdm.auto import tqdm

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
# %%
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor, WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from tokenizers.processors import TemplateProcessing
from hubert_linear import HubertLinear 
from whisper_linear import WhisperLinear 
from wav2vec_linear import Wav2VecLinear, Wav2VecLoRA
from transformers import HubertModel, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from transformers import WhisperModel, Wav2Vec2Model


os.environ["WANDB_DISABLED"] = "true"

parser = ArgumentParser()
## model args
parser.add_argument('--model_name', type=str, default='wav2vec2-base', help='Name of the pre-trained model')
parser.add_argument('--model_path', type=str, default='../outputs', help='saved model path')
parser.add_argument('--data_path', type=str, default='../datasets/data_phonem/',)
parser.add_argument('--is_whisper', action='store_true',)
parser.add_argument('--is_hubert', action='store_true',)
parser.add_argument('--is_wembed', action='store_true',)

parser.add_argument('--is_lora', action='store_true', help='Use LoRA model')


parser.add_argument('--subject', type=str, default='3', help='Subject number to train on')


parser.add_argument('--out_dim', type=int, default=95556, help='Output dimension of the linear layer')
parser.add_argument('--nc_thr', type=float, default=0.4, help='Learning rate for the linear layer')

parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
parser.add_argument('--num_epochs', type=int, default=100)
## data args
parser.add_argument('--sampling_rate', type=int, default=16000, help='Sampling rate for the wav file')


## logs args
parser.add_argument('--save_dir', type=str, default='../outputs/senttype_preds')
parser.add_argument('--model_suf', type=str, default='eval_sent_wav2vec', help='Model suffix')
parser.add_argument('--exp_name', type=str, default='eval_sent_wav2vec', help='Experiment name')

args = parser.parse_args()

timit_path = args.data_path
data_path = args.data_path
device = args.device

exp_name = 'senttype'
if args.is_hubert:
    model_name = "facebook/hubert-base-ls960"

elif args.is_whisper:
    model_name = 'openai/whisper-small'
else:
    model_name = 'facebook/wav2vec2-base'
    
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
        lm = Wav2Vec2Model.from_pretrained(f'facebook/{model_name}')
    else:
        raise ValueError('Model type not found')
    
    return lm


timit = load_dataset(path="timit_asr", data_dir='../datasets', cache_dir="../datasets")
train_dataset = timit["train"]
test_dataset = timit["test"]
# edict = load_glove_embeddings()
layers = [2, 5, 7, 8, 10, 11, 12]
# layers = [2, 12, 24, 36, 44, 48] #[2, 5, 7, 8, 10, 11, 12]

# layers = [2, 7, 12, 16, 20, 24] #[2, 5, 7, 8, 10, 11, 12]

vocab_dict = json.load(open(f'{data_path}/vocab.json'))


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
    
# # %%
# tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(f"{data_path}/", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", )  # './' load vocab.json in the current directory
# feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)  
# processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
type2lbl = dict(SA=0, SI=1, SX=2)
lbl2type = {0:'SA', 1:'SA', 2:'SX'}


def prepare_dataset(batch):
    audio = batch["audio"]
    
    # batched output is "un-batched"
    
    if args.is_whisper:
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    else:
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        
    batch["input_length"] = len(batch["input_values"])
    
    batch["labels"] = (type2lbl[batch["sentence_type"]])
    return batch


train_dataset = train_dataset.map(prepare_dataset, remove_columns=['file', 'audio', 'text', 'phonetic_detail', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'], num_proc=4)
test_dataset = test_dataset.map(prepare_dataset, remove_columns=['file', 'audio', 'text', 'phonetic_detail', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'], num_proc=4)

best_mean = args.model_path
model_suf = args.model_suf
save_root = f'{args.data_path}/phonem/'





if 'base' in (args.subject):
    if 'ssl' in (args.subject):
        print('loading SSL model from ckpt {}'.format(best_mean))
        lm_b = load_ssl_model(args.model_path)
    else:
        lm_b = init_base_models(model_name)

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
    ckpt = (torch.load(best_mean))
    # if 'module.' in list(ckpt.keys())[0]:
    #     ## for data parallel model
    #     ckpt = {key.replace('module.wav2vec.', ''): value for key, value in ckpt.items()}
    
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


    lm.eval()
    lm = lm.to(device)
    

if 'base' not in args.subject:
    print('loading model from ckpt {}'.format(best_mean))
    lm.load_state_dict(ckpt_w)

    if args.is_whisper:
        lm_b = lm.whisper_encoder 
    elif args.is_hubert:
        lm_b = lm
    else:
        lm_b = lm
        # lm_b = lm.wav2vec
    
        

lm_b.eval(); lm_b = lm_b.to(device)



features_dict_tr, labels_tr = get_probing_data(train_dataset, lm_b, layers, device)
save_word_data(features_dict_tr, labels_tr, layers, subject=args.subject, phase='train', model_suf=model_suf, save_root=save_root, exp_name=exp_name)

features_dict, labels = get_probing_data(test_dataset, lm_b, layers, device)
save_word_data(features_dict, labels, layers, subject=args.subject, phase='test', model_suf=model_suf, save_root=save_root, exp_name=exp_name)

            

per_ly_preds = {}
for layer in layers:
    phase = 'train'
    Xtr =  np.load(f"{save_root}/{exp_name}_{model_suf}_{args.subject}_{phase}_layer{layer}_features.npy")
    ytr = np.load(f"{save_root}/{exp_name}_{model_suf}_{phase}_labels.npy",)

    phase = 'test'
    Xt =  np.load(f"{save_root}/{exp_name}_{model_suf}_{args.subject}_{phase}_layer{layer}_features.npy")
    yt = np.load(f"{save_root}/{exp_name}_{model_suf}_{phase}_labels.npy",)
    print(f'layer {layer} process, training shapes: {Xtr.shape}, {ytr.shape}, testing shapes: {Xt.shape}, {yt.shape}')
    p = train_layer_senttype_model(Xtr, ytr, Xt, yt, num_epochs=args.num_epochs, device=device, num_features=Xtr.shape[1],)
    per_ly_preds[layer] = p

## save preds
if args.is_wembed:
    save_dst = f"{args.save_dir}/LLama2_7b"

else:
    save_dst = f"{args.save_dir}/UTS0{args.subject}"


os.makedirs(save_dst, exist_ok=True)    

with open(f"{save_dst}/{model_suf}_preds.pkl", 'wb') as f:
    pickle.dump(per_ly_preds, f)
# %%
# f1_scores = []
# for ly in layers:
#     f1_scores.append(per_ly_preds[ly][49][-1])
    
# f1_mean = []
# for ly in layers:
#     f1_mean.append(per_ly[ly][49][-1])
