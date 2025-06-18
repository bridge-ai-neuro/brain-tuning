import torch
from torch import nn, optim
from transformers import WhisperModel, WhisperProcessor, AutoProcessor
from transformers import WhisperFeatureExtractor


class WhisperLinear(nn.Module):
    def __init__(self,
                 out_dim,
                 model_name='openai/whisper-small',
                 pooling_strategy='mean',
                 sampling_rate=16000,
                 output_hidden_states=False):
        super(WhisperLinear, self).__init__()
        
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(f"{model_name}")
        
        self.whisper_model = WhisperModel.from_pretrained(f'{model_name}')
        ## freze the decoder 
        for param in self.whisper_model.decoder.parameters():
            param.requires_grad = False
        
        self.whisper_encoder = self.whisper_model.encoder # train only the encoder from the Whisper model
        
        self.processor = AutoProcessor.from_pretrained(f"{model_name}", return_tensors="pt")
        self.output_hidden_states = output_hidden_states
        self.non_trainables = ['conv1', 'conv2', 'embed_positions'] # freeze the feature enc. 
        
        for pr in self.non_trainables:
            if getattr(self.whisper_encoder, pr, None):
                for param in getattr(self.whisper_encoder, pr).parameters():
                    param.requires_grad = False
        # print(self.wav2vec.config.hidden_size)
        for name, param in self.whisper_encoder.named_parameters():
            if not param.requires_grad:
                print(f"Layer {name} is frozen.")
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.whisper_model.config.hidden_size, out_dim)
        self.pooling_strategy = pooling_strategy
        self.sampling_rate = sampling_rate
    
    def forward(self, input_wav_tensor):
        # with torch.no_grad():  # Optionally freeze wav2vec to prevent fine-tuning its weights
        tensor_device = input_wav_tensor.device
        bsize = len(input_wav_tensor)
        attn_mask = self.feature_extractor(input_wav_tensor[0].cpu().numpy(), return_tensors="pt", return_attention_mask=True, sampling_rate=self.sampling_rate)['attention_mask']
        
        out_idx = self._get_indices(attn_mask) # get idx of last effective input
        ## input processing
        processed_output = torch.cat([self.feature_extractor(wav_slice.cpu().numpy(), return_tensors="pt", sampling_rate=self.sampling_rate).input_features for wav_slice in input_wav_tensor],
                                     dim=0).to(tensor_device)
        extracted_features = self.whisper_encoder(processed_output, self.output_hidden_states).last_hidden_state
        hidden_states = self.merged_strategy(extracted_features[:,:out_idx], mode=self.pooling_strategy) # pass only the effective inputs
        output = self.linear(hidden_states)
        return output

    def get_n_params(self, model):
        pp=0
        for p in list(model.parameters()):
            if not p.requires_grad: continue
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    def _get_trainable_params(self):
        ## Assert # trainable params 
        total = self.get_n_params(self); comp = self.get_n_params(self.whisper_encoder) + self.get_n_params(self.linear)
        print(f"Total Trainable Params : {total/1e6}M")
        print(f"Computed Ind Total Trainable Params: {comp/1e6}M ")
        assert total == comp, f"Total params mismatch, {total - comp}"

    
    def _get_indices(self, attn_masks):
        
        contributing_outs = attn_masks[0].unsqueeze(0)

        contributing_outs = torch.nn.functional.conv1d(contributing_outs,
                                                        torch.ones((1,1)+self.whisper_encoder.conv1.kernel_size).to(contributing_outs),
                                                        stride=self.whisper_encoder.conv1.stride,
                                                        padding=self.whisper_encoder.conv1.padding,
                                                        dilation=self.whisper_encoder.conv1.dilation,
                                                        groups=self.whisper_encoder.conv1.groups)
        # shape: (batchsz, 1500)
        contributing_outs = torch.nn.functional.conv1d(contributing_outs,
                                                        torch.ones((1,1)+self.whisper_encoder.conv2.kernel_size).to(contributing_outs),
                                                        stride=self.whisper_encoder.conv2.stride,
                                                        padding=self.whisper_encoder.conv2.padding,
                                                        dilation=self.whisper_encoder.conv2.dilation,
                                                        groups=self.whisper_encoder.conv1.groups)
        
        final_output = contributing_outs[0].nonzero().squeeze(-1).max()
        return final_output.detach().item()
        
    
        
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
        return list(filter(lambda p: p.requires_grad, self.whisper_encoder.parameters()))



class SimpleLinearModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.linear(x)
    
