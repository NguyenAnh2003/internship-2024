from transformers import XLMRobertaModel, AutoTokenizer
from omegaconf import DictConfig, OmegaConf
from torch.nn import (Module, Sequential, LSTM, Dropout,
                      BatchNorm1d, Linear, LayerNorm)
import torch.nn as nn
import torch
from libs.helper_functions import get_configs

class RNNModule(Module):
    def __init__(self, input_size, hidden_size):
        super(RNNModule, self).__init__()
        self.lstm = LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return out


class ABSAModel(nn.Module):
    def __init__(self, conf: DictConfig, input_size=768,
                 hidden_size=384, output_size=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf = OmegaConf.create(conf)

        # Initialize model - encoder
        self.encoder, self.tokenizer = self._init_pretrained_metadata()

        # Convolutional layer
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3, padding=1)

        # RNN block
        self.rnn_block = RNNModule(input_size=input_size, hidden_size=hidden_size)  # in_size, hidden_size

        # Norm
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
        rep = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = rep.last_hidden_state  # B, L, D

        # Apply Conv1D layer
        last_hidden_state = last_hidden_state.permute(0, 2, 1)  # B, D, L
        conv_out = self.conv1d(last_hidden_state)  # B, D, L
        conv_out = conv_out.permute(0, 2, 1)  # B, L, D

        # Pass through RNN
        rnn_out = self.rnn_block(conv_out)  # B, L, D

        # Skip connection
        rnn_out = rnn_out + conv_out
        out = self.norm(rnn_out)
        out = self.classifier(out[:, -1, :])
        logtis = self.log_softmax(out)
        return logtis


if __name__ == "__main__":
    conf = get_configs("../configs/absa_model.yaml")
    conf["model"]["pretrained"]["name"] = "FacebookAI/xlm-roberta-base"
    conf["model"]["pretrained"]["freeze"] = True
    model = ABSAModel(conf)
    print(model)