import torch
import torch.nn as nn
from transformers import HubertModel, Wav2Vec2Processor, Wav2Vec2FeatureExtractor

class HubertLinear(nn.Module):
    def __init__(self,
                 out_dim,
                 model_name="facebook/hubert-base-ls960",
                 return_hidden=False,
                 pooling_strategy='mean',
                 sampling_rate=16000,):
        super(HubertLinear, self).__init__()
        self.return_hidden = return_hidden
        print('loading model', model_name)
        self.hubert = HubertModel.from_pretrained(model_name)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, return_tensors="pt")
        self.hubert.feature_extractor._freeze_parameters()
        # print(self.wav2vec.config.hidden_size)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.hubert.config.hidden_size, out_dim)
        self.pooling_strategy = pooling_strategy
        self.sampling_rate = sampling_rate
    
    def forward(self, input_wav_tensor):
        # with torch.no_grad():  # Optionally freeze wav2vec to prevent fine-tuning its weights
        tensor_device = input_wav_tensor.device
        bsize = len(input_wav_tensor)
        processed_output = self.processor(input_wav_tensor, return_tensors="pt", sampling_rate=self.sampling_rate).input_values.squeeze().reshape(bsize, -1).to(tensor_device)
        extracted_features = self.hubert(processed_output, output_hidden_states=self.return_hidden).last_hidden_state
        hidden_states = self.merged_strategy(extracted_features, mode=self.pooling_strategy)
        # print(extracted_features)  # [batch_size, seq_len, hidden_size
        # flat_features = self.flatten(extracted_features)
        # avg_features = torch.mean(extracted_features, dim=1)# [batch_size, hidden_size]
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

        return outputs

    def get_trainable_parameters(self):
        return list(filter(lambda p: p.requires_grad, self.hubert.parameters()))


class SimpleLinearModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.linear(x)
    
# Note: You may need to adjust input shapes and dataloader according to your specific needs.
