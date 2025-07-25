import torch
import torch.nn as nn
import torch.optim as optim
from wav2vec_linear import Wav2VecLinear
from hubert_linear import HubertLinear
from whisper_linear import WhisperLinear
from argparse import ArgumentParser
from dataset import *
from tqdm import tqdm
import numpy as np 
import time
import pickle
from torch.utils.data import Dataset, DataLoader

parser = ArgumentParser()
## model fine-tuning params
parser.add_argument('--model_name', type=str, default='facebook/wav2vec2-base', help='Name of the pre-trained model')
parser.add_argument('--out_dim', type=int, default=95556, help='Output dimension of the linear layer')
parser.add_argument('--base_lr', type=float, default=5e-5, help='Learning rate for the base model')
parser.add_argument('--linear_layer_lr', type=float, default=1e-4, help='Learning rate for the linear layer')
parser.add_argument('--nc_thr', type=float, default=0.4, help='Learning rate for the linear layer')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs for training')
parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')

## Dataset params
parser.add_argument('--subject', type=int, default=3, help='Subject number to train on')
parser.add_argument('--sampling_rate', type=int, default=16000, help='Sampling rate for the wav file')
parser.add_argument('--chunksz_sec', type=int, default=1, help='Chunk size in seconds')
parser.add_argument('--contextsz_sec', type=int, default=0, help='Context size in seconds')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')

## logs args
parser.add_argument('--save_dir', type=str, default='../outputs/train_logs_2')
parser.add_argument('--save_name', type=str, default='mean_wav2vec')
parser.add_argument('--exp_name', type=str, default='wav2vec_story_mp')


args = parser.parse_args()
total_epochs = args.num_epochs  # Total number of epochs for training
increase_epochs = 5  # Epochs to reach the peak learning rate
def schedule_group_0(epoch):
    if epoch < increase_epochs:
        return epoch / increase_epochs
    else:
        return 1 - (epoch - increase_epochs) / (total_epochs - increase_epochs)

# Schedule for the second parameter group (exponential decay)
def schedule_group_1(epoch):
    return max(0.8 ** epoch, 0.1)

# Combining the schedules
def combined_schedule(epoch):
    return [schedule_group_0(epoch), schedule_group_1(epoch)]

def evaluate(model, dataloader, loss_function, device):
    # model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_wav_tensor, output_signal in dataloader:
            input_wav_tensor = input_wav_tensor.to(device)
            output_signal = output_signal.to(device)
            predictions = model(input_wav_tensor)
            loss = loss_function(predictions, output_signal)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def init_model(out_dim, model_name, sampling_rate, device):
    
    if 'wav2vec' in model_name:
        model = Wav2VecLinear(out_dim=out_dim,
                              model_name=model_name, # default is 'facebook/wav2vec2-base',
                              sampling_rate=sampling_rate).to(device)
    elif 'hubert' in model_name:
        model = HubertLinear(out_dim=out_dim,
                             model_name=model_name, # default is 'facebook/hubert-base-ls960',
                             sampling_rate=sampling_rate).to(device)
    elif 'whisper' in model_name:
        model = WhisperLinear(out_dim=out_dim,
                              model_name=model_name, # default is 'openai/whisper-small',
                              sampling_rate=sampling_rate).to(device)
    else:
        raise ValueError(f"Model {model_name} is not supported. Choose from 'wav2vec', 'hubert', or 'whisper'.")
    return model

if __name__ == '__main__':
    
    # define noise ceiling mask and wav params
    sub_nc = np.load(f'subject_NCs/UTS0{args.subject}.npy')
    sub_neur_mask = np.where(sub_nc > args.nc_thr)[0]
    out_dim = len(sub_neur_mask)  # Desired output dimension based on noise ceiling mask
    device = args.device
    wav_params = {'sampling_rate': args.sampling_rate, 'chunksz_sec': args.chunksz_sec, 'contextsz_sec':args.contextsz_sec}
    
    # Initialize the dataset and dataloader

    train_dataloaders = []
    for t_story in train_stories:
        story_ds = FMRIStory(story_name=t_story, subject=args.subject, sub_nc_mask=sub_neur_mask, **wav_params) #hangtime
        story_dl = DataLoader(story_ds, batch_size=args.batch_size, shuffle=False)
        train_dataloaders.append(story_dl)
    
    test_ds = FMRIStory(story_name=val_stories[0], subject=args.subject, sub_nc_mask=sub_neur_mask, **wav_params)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    test_dataloader.dataset.fetch_data()

    ## Initialize the model
    model = init_model(out_dim=out_dim, model_name=args.model_name, sampling_rate=args.sampling_rate, device=device)
    num_epochs = args.num_epochs


    # Set learning rates
    base_lr = args.base_lr
    linear_layer_lr = args.linear_layer_lr
    trainable_params = model.get_trainable_parameters()
    # Prepare optimizer with different learning rates
    optimizer = optim.Adam([
        {'params': trainable_params, 'lr': base_lr},
        {'params': model.linear.parameters(), 'lr': linear_layer_lr}
    ])

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[schedule_group_0, schedule_group_1])

    ## model training 
    model = torch.nn.DataParallel(model) # for multi-GPU training
    # Loss function
    loss_function = nn.MSELoss()

    t = time.time()
    # Training loop
    losses_dict = dict(train=[], test=[])
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = [] 
        for s_idx, t_story in enumerate(tqdm((train_stories_25))):
            
            trainloader = train_dataloaders[s_idx]
            trainloader.dataset.fetch_data() # fetch data for the current story
            story_loss = []
            for input_wav_tensor, output_signal in (trainloader):
                
                optimizer.zero_grad()
                input_wav_tensor = input_wav_tensor.to(device)
                output_signal = output_signal.to(device)
                predictions = model(input_wav_tensor)
                loss = loss_function(predictions, output_signal)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item()); story_loss.append(loss.item())
            
            trainloader.dataset._clear() # free memory after each story
            print(f'Epoch {epoch+1}, Story {t_story}, story Train Loss: {np.mean(story_loss)}')
            story_loss = []
            
        
        eval_loss = evaluate(model, test_dataloader, loss_function, device=device)
        print(f'Epoch {epoch+1}, Train Loss: {np.mean(epoch_loss)}, Eval Loss: {eval_loss}')
        losses_dict['train'].append(np.mean(epoch_loss)); losses_dict['test'].append(eval_loss)
        epoch_loss = []
        scheduler.step()
        if epoch % 5 == 0 or epoch <= 5:
            torch.save(model.state_dict(), f'{args.save_dir}/{args.exp_name}/{args.subject}_{args.save_name}_epoch_{epoch}.pth')
    
    # Save the model
    print(f'Training took {time.time() - t} seconds')
    torch.save(model.state_dict(), f'{args.save_dir}/{args.exp_name}/{args.subject}_{args.save_name}_epochs_{args.num_epochs}.pth')
    pickle.dump(losses_dict, open(f'{args.save_dir}/logs/{args.exp_name}/{args.subject}_{args.save_name}_losses.pkl', 'wb'))
    