from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    pipeline
from omegaconf import OmegaConf, DictConfig
from transformers import Trainer, TrainingArguments
from data_manipulation.dataloader import ABSADataset
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

    def get_model_parameters(self):
        params = sum([i for i in self.model.parameters()])
        return params # sum of parameters

    def setup_dataset(self, path):
        absa = ABSADataset(tokenizer=self.tokenizer, csv_path=path, conf=self.conf)
        dataset = absa.dataset # HF dataset

        return dataset

    def finetuning_pretrained_model(self):

        if self.conf.model.pretrained.freeze == True:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

        print(f"Model: {self.model} " f"Tokenizer: {self.tokenizer}")

