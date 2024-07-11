from transformers import AutoModelForSequenceClassification, AutoTokenizer
from omegaconf import OmegaConf, DictConfig
import torch.nn.functional as F
from libs.helper_functions import get_configs

class PModelModule:
    def __init__(self, conf: DictConfig = None) -> None:
        self.conf = OmegaConf.create(conf)

        # get pretrained model
        self.model, self.tokenizer = self.get_pretrained_model_and_tokenizer()

    def get_pretrained_model_and_tokenizer(self):
        # using transformers package to get pretrained model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.conf.model.pretrained.name
        )  # get pretrained model name

        tokenizer = AutoTokenizer.from_pretrained(self.conf.model.pretrained.name)
        model.eval()  #

        return model, tokenizer


    def _setup_train_dataloader(self, path):
        pass

    def finetuning_pretrained_model(self):

        if self.conf.model.pretrained.freeze == True:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

        print(f"Model: {self.model} " f"Tokenizer: {self.tokenizer}")
        
