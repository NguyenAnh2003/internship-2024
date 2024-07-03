from libs.helper_functions import get_configs
from models.model import ABSAModel

if __name__ == "__main__":
  conf = get_configs("../configs/absa_dp.yaml")

  # multi lingual model included Thai and En
  conf["model"]["pretrained"]["name"] = "google-bert/bert-base-multilingual-cased"

  model = ABSAModel(conf)

  sample = "Hello my name is"

  representation = model._embedding_of_input(sample)

  print(f"Parameters: {model.get_model_params()} "
        f"JJJ: {representation.shape}")
