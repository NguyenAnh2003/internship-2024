from transformers import XLMRobertaModel, AutoTokenizer
from omegaconf import DictConfig, OmegaConf
from torch.nn import (Module, Sequential, LSTM, Dropout,
                      BatchNorm1d, Linear, LayerNorm)
import torch.nn as nn
import torch
from libs.helper_functions import get_configs
import numpy as np

class RNNModule(Module):
    def __init__(self, input_size, hidden_size):
        super(RNNModule, self).__init__()
        self.lstm = LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.2)
        self.norm = nn.LayerNorm(normalized_shape=input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out)
        out = self.dropout(out)
        return out

class CNNModule(nn.Module):
    def __init__(self, input_size, conv_out_channels, kernel_size=3, dropout_p=0.2):
        super(CNNModule, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=conv_out_channels, kernel_size=kernel_size, padding=1)
        self.norm = nn.LayerNorm(normalized_shape=conv_out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        # Apply normalization
        x = x.permute(0, 2, 1)  # Change shape to [batch, length, channels]
        x = self.norm(x)  # Apply LayerNorm
        x = x.permute(0, 2, 1)  # Change back to [batch, channels, length]
        # Apply ReLU activation
        x = self.relu(x)
        # Apply dropout
        x = self.dropout(x)
        return x

class ABSAModel(nn.Module):
    def __init__(self, conf: DictConfig, input_size=768,
                 hidden_size=384, output_size=10, num_cnn_layers=1, num_rnn_layers=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf = OmegaConf.create(conf)

        # Initialize model - encoder
        self.encoder, self.tokenizer = self._init_pretrained_metadata()

        # Convolutional layers with normalization, ReLU, and dropout
        self.cnn_layers = nn.ModuleList([
            CNNModule(input_size=input_size,
                      conv_out_channels=input_size)
            for i in range(num_cnn_layers)
        ])

        # RNN layers with residual connections
        self.rnn_layers = nn.ModuleList([
            RNNModule(input_size=input_size,
                      hidden_size=hidden_size)
            for _ in range(num_rnn_layers)
        ])

        # Normalization
        self.norm = nn.LayerNorm(normalized_shape=hidden_size * 2)

        # Initialize MLP
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, output_size)
        )

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def _init_pretrained_metadata(self):
        model = XLMRobertaModel.from_pretrained(self.conf.model.pretrained.name)
        tokenizer = AutoTokenizer.from_pretrained(self.conf.model.pretrained.name)

        if self.conf.model.pretrained.freeze:
            for params in model.parameters():
                params.requires_grad = False

        return model, tokenizer

    # Getting embedding of each input
    def get_model_params(self):
        params = [p.nelement() for p in self.encoder.parameters()]
        return sum(params)

    def forward(self, input_ids, attention_mask):
        # Encode the input
        rep = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = rep.last_hidden_state  # Shape: [B, L, D]
        res = last_hidden_state
        # Apply CNN layers
        cout = last_hidden_state.permute(0, 2, 1)  # Shape: [B, D, L]
        for cnn in self.cnn_layers:
            cout = cnn(cout)  # Shape: [B, C, L]

        conv_out = cout.permute(0, 2, 1)  # Shape: [B, L, C]

        # Add the original hidden state to CNN output (residual connection)
        out = conv_out + res # Shape: [B, L, D] after addition

        # Apply RNN layers with residual connections
        residual = out
        for rnn in self.rnn_layers:
            out = rnn(residual)
            if out.size(1) == residual.size(1):  # Ensure the sequence length is the same
                out = out + residual  # Add residual connection
            residual = out  # Update residual for the next layer

        # Apply normalization and classifier
        out = self.norm(out)  # Normalize
        out = self.classifier(out[:, -1, :])  # Use the last time step's output for classification
        logtis = self.log_softmax(out)  # Apply LogSoftmax

        return logtis


if __name__ == "__main__":
    conf = get_configs("../configs/absa_model.yaml")
    conf["model"]["pretrained"]["name"] = "FacebookAI/xlm-roberta-base"
    conf["model"]["pretrained"]["freeze"] = True
    model = ABSAModel(conf)
    print(model)