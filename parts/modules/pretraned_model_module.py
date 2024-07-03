from transformers import AutoModel, BertModel, BertTokenizer
from omegaconf import OmegaConf, DictConfig


class ModelModule:
    def __init__(self, conf: DictConfig = None) -> None:
        self.conf = OmegaConf.create(conf)

        # get pretrained model
        self.model, self.tokenizer = self.get_pretrained_model_and_tokenizer()

    def get_pretrained_model_and_tokenizer(self):

        # using transformers package to get pretrained model
        if self.conf.model.pretrained.bert == False:
            model = AutoModel.from_pretrained(
                self.conf.model.pretrained.name
            )  # get pretrained model name
        else:
            model = BertModel.from_pretrained(
                self.conf.model.pretrained.name
            )

            tokenizer = BertTokenizer.from_pretrained(
                self.conf.model.pretrained.name
            )

        model.eval() #

        return model, tokenizer

    def finetuning_pretrained_model(self):

        if self.conf.model.pretrained.freeze == True:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

        print(f"Model: {self.model} "
              f"Tokenizer: {self.tokenizer}")