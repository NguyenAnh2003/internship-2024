from transformers import XLMRobertaModel, AutoTokenizer
from omegaconf import DictConfig, OmegaConf
from torch.nn import (Module, Sequential, LSTM, Dropout,
                      BatchNorm1d, Linear)
import torch.nn as nn
import torch
from libs.helper_functions import get_configs

class RNNModule(Module):
    def __init__(self, input_size, hidden_size):
        super(RNNModule, self).__init__()
        self.lstm = LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.norm = BatchNorm1d(num_features=hidden_size * 2)  # For bidirectional
        self.dropout = Dropout(p=0.1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.permute(0, 2, 1)  # For batch normalization
        out = self.norm(out)
        out = out.permute(0, 2, 1)
        out = self.dropout(out)
        return out


class ABSAModel(Module):
    def __init__(self, conf: DictConfig, input_size = 768,
                 hidden_size = 256, output_size = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf = OmegaConf.create(conf)

        # init model - encoder
        self.encoder, self.tokenizer = self._init_pretrained_metadata()

        # rnn block
        self.rnn_block = RNNModule(input_size=input_size, hidden_size=hidden_size) # in_size, hidden_size

        # init mlp
        self.classifier = Sequential(Linear(hidden_size*2, hidden_size),
                                     nn.ReLU(),
                                     Dropout(0.1),
                                     Linear(hidden_size, output_size))

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def _init_pretrained_metadata(self):
        model = XLMRobertaModel.from_pretrained(self.conf.model.pretrained.name)
        tokenizer = AutoTokenizer.from_pretrained(self.conf.model.pretrained.name)

        if self.conf.model.pretrained.freeze == True:
            for params in model.parameters():
                params.requires_grad = False

        return model, tokenizer

    # getting embedding of each input
    def get_model_params(self):
        params = [p.nelement() for p in self.pretrained_model.parameters()]
        return sum(params)

    def forward(self, input_ids, attention_mask):
        rep = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = rep.last_hidden_state
        rnn_out = self.rnn_block(last_hidden_state)
        out = self.classifier(rnn_out)
        logtis = self.log_softmax(out)
        return logtis

if __name__ == "__main__":
    conf = get_configs("../configs/absa_model.yaml")
    conf["model"]["pretrained"]["name"] = "FacebookAI/xlm-roberta-base"
    conf["model"]["pretrained"]["freeze"] = True
    model = ABSAModel(conf)
    print(model)