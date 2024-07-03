from parts.modules.pretraned_model_module import ModelModule
from omegaconf import DictConfig, OmegaConf
from torch.nn import Module, Sequential
import torch

class ABSAModel(Module):
    def __init__(self, conf: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf = OmegaConf.create(conf)

        # define pretrained model module
        self.module = ModelModule(conf) # define pretrained model module
        self.pretrained_model = self.module.model # pretrained model

        if self.conf.model.pretrained.freeze == True:
            for params in self.pretrained_model.parameters():
                params.requires_grad = False

        self.tokenizer = self.module.tokenizer # pretrained model's tokenizer

    # getting embedding of each input
    def _embedding_of_input(self, input: str):
        tokenized_text = self.tokenizer.tokenize(input)
        segment_ids = self.tokenizer.convert_tokens_to_ids(input)
        segment_ids = [1] * len(tokenized_text)
        segment_tensor = torch.tensor([segment_ids])

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            outputs = self.pretrained_model(tokens_tensor, segment_tensor)
            last_hidden_state = outputs.last_hidden_state

        return last_hidden_state

    def get_model_params(self):
        params = [p.nelement() for p in self.pretrained_model.parameters()]
        return sum(params)

    def forward(self, x):

        return