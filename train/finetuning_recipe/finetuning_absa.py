from parts.modules.pretraned_model_module import ModelModule
from libs.helper_functions import get_configs
from transformers import Trainer, TrainingArguments


if __name__ == "__main__":
    conf = get_configs("../configs/absa_dp.yaml")
    conf["model"]["pretrained"]["name"] = "google-bert/bert-base-multilingual-cased"
    conf["model"]["pretrained"]["freeze"] = True

    model = ModelModule(conf)
    model.finetuning_pretrained_model()
