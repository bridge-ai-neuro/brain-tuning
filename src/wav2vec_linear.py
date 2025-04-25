import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2ForPreTraining

class Wav2VecLinear(nn.Module):
    def __init__(self, out_dim,
                 wav2vec_model_name='facebook/wav2vec2-base',
                 pooling_strategy='mean',
                 sampling_rate=16000,):
        super(Wav2VecLinear, self).__init__()
        print(f'loading wav2vec model with {wav2vec_model_name}')
        self.wav2vec = Wav2Vec2Model.from_pretrained(f'{wav2vec_model_name}')
        self.processor = Wav2Vec2Processor.from_pretrained(f"{wav2vec_model_name}", return_tensors="pt")
        self.wav2vec.freeze_feature_encoder()
        # print(self.wav2vec.config.hidden_size)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.wav2vec.config.hidden_size, out_dim)
        self.pooling_strategy = pooling_strategy
        self.sampling_rate = sampling_rate
    
    def forward(self, input_wav_tensor):
        # with torch.no_grad():  # Optionally freeze wav2vec to prevent fine-tuning its weights
        tensor_device = input_wav_tensor.device
        bsize = len(input_wav_tensor)
        processed_output = self.processor(input_wav_tensor, return_tensors="pt", sampling_rate=self.sampling_rate).input_values.squeeze().reshape(bsize, -1).to(tensor_device)
        extracted_features = self.wav2vec(processed_output).last_hidden_state
        hidden_states = self.merged_strategy(extracted_features, mode=self.pooling_strategy)

        output = self.linear(hidden_states)
        return output

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

 
class SimpleLinearModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.linear(x)
    
