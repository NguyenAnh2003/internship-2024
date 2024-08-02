from transformers import AutoModel, AutoTokenizer
from omegaconf import DictConfig, OmegaConf
from torch.nn import (Module, Sequential, LSTM, Dropout,
                      BatchNorm1d, Linear)
import torch.nn as nn
import torch
from libs.helper_functions import get_configs

class RNNModule(Module):
    def __init__(self, input_size, hidden_size):
        super.__init__(RNNModule, self)
        self.lstm = LSTM(input_size, hidden_size,
                         batch_first=True, bidirectional=True)
        self.norm = BatchNorm1d(num_features=input_size)
        self.dropout = Dropout(p=0.1)

    def forward(self, x):
        out, hidden = self.lstm(x)
        out = self.norm(out)
        out = self.dropout(x)
        return out


class ABSAModel(Module):
    def __init__(self, conf: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf = OmegaConf.create(conf)

        # init model - encoder
        self.model, self.tokenizer = self._init_pretrained_metadata()

        # rnn block
        # self.rnn_block = RNNModule() # in_size, hidden_size
        #
        # # init mlp
        # self.classifier = Sequential(Linear(),
        #                              nn.ReLU(),
        #                              Dropout(),
        #                              Linear())
    def _init_pretrained_metadata(self):
        model = AutoModel.from_pretrained(self.conf.model.pretrained.name)
        tokenizer = AutoTokenizer.from_pretrained(self.conf.model.pretrained.name)

        if self.conf.model.pretrained.freeze == True:
            for params in model.parameters():
                params.requires_grad = False

        return model, tokenizer

    # getting embedding of each input
    def get_model_params(self):
        params = [p.nelement() for p in self.pretrained_model.parameters()]
        return sum(params)

    def forward(self, x):
        representation = self.model(**x)
        return

if __name__ == "__main__":
    conf = get_configs("../configs/absa_model.yaml")
    print(conf)
    conf["model"]["pretrained"]["name"] = "FacebookAI/xlm-roberta-base"
    conf["model"]["pretrained"]["freeze"] = True
    model = ABSAModel(conf)