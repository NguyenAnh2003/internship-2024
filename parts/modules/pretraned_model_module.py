from transformers import AutoModel
from omegaconf import OmegaConf, DictConfig


class ModelModule:
    def __init__(self, conf: DictConfig = None) -> None:
        self.conf = OmegaConf.create(conf)

        # get pretrained model
        self.model = self.get_pretrained_model()

        # get tokenizer
        self.tokenizer = self.get_tokenizer()

    def get_pretrained_modelntokenizer(self):
        # using transformers package to get pretrained model
        model = AutoModel.from_pretrained(
            self.pretrained.name
        )  # get pretrained model name
        return model

    def get_tokenizer(self):
        tokenizer = None

        return tokenizer