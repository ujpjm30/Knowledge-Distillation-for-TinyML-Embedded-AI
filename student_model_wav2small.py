import torch
from torch import nn
from torchlibrosa import Spectrogram, LogmelFilterBank
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Tuple
import os
import json

# def conv(in=13):
#     return nn.Sequential(nn.Conv2d(in,13,k=3,pad=1, bias=False), nn.BatchNorm2d(13), nn.ReLU())
@dataclass
class SpeechClassifierObject(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class ModelOutput:
    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits

class Wav2Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg7 = nn.Sequential(
            Spectrogram(n_fft=64, hop_length=32),
            LogmelFilterBank(n_fft=64, n_mels=26),
            self._conv(in_channels=1),
            self._conv(in_channels=13),
            self._conv(in_channels=13),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(0.2),
            self._conv(in_channels=13),
            self._conv(in_channels=13),
            self._conv(in_channels=13),
            self._conv(in_channels=13),
            nn.Dropout(0.2),
            nn.Conv2d(13, 13, kernel_size=1)
        )
        self.classifier = nn.Linear(169, 7)  # 7 TESS emotions

        self._initialize_weights()
        
    def _conv(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 13, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(13),
            nn.ReLU()
        )
    
    def forward(self, input_values, labels=None):
        # Input normalization
        input_values = input_values - input_values.mean(dim=1, keepdim=True)
        variance = (input_values * input_values).mean(dim=1, keepdim=True) + 1e-7
        input_values = input_values / torch.sqrt(variance)
        
        # Feature extraction
        input_values = self.vgg7(input_values)
        batch, channel, token, mel = input_values.shape
        
        # Reshape tokens for classification
        input_values = input_values.reshape(batch, token, channel * mel)
        input_values = torch.mean(input_values, dim=1)  # Global average pooling
        
        # Classification
        logits = self.classifier(input_values)

        # Loss computation
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 7), labels.view(-1))
            
        return SpeechClassifierObject(
            loss=loss,
            logits=logits,
        )

    def save_pretrained(self, save_directory):
        state_dict = self.state_dict()
        for key, param in state_dict.items():
            if not param.is_contiguous():
                state_dict[key] = param.contiguous()

        torch.save(state_dict, os.path.join(save_directory, 'pytorch_model.bin'))

        config = {
            'n_fft': 64,
            'hop_length': 32,
            'n_mels': 26,
            'num_labels': 7
        }
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(config, f)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)