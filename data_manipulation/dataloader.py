from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datasets import Dataset as HFDataset
from omegaconf import OmegaConf, DictConfig
from libs.helper_functions import get_configs


class ABSADataset(Dataset):
    def __init__(self, tokenizer=None, csv_path="", conf: DictConfig = None) -> None:
        super().__init__()
        self.conf = OmegaConf.create(conf)
        self.tokenizer = tokenizer  # tokenizer as transform
        self.dataset = self._create_hf_ds(csv_path=csv_path)  # create HF dataset
        self.dataset = self.dataset.map(
            self._instruction_template4train
        )  # mapping data

    def __getitem__(self, index):
        pass

    def _instruction_template4train(self, batch):
        # prepare HF dataset
        review = batch["review"]
        aspect = batch["aspect"]  # label2index
        opinion = batch["opinion"]  # tokenize this
        polarity = batch["polarity"]  # label2index
        prompt = ""
        # take review as input and aspect, polarity as output
        if self.conf.model.instruction_style == "simple":
            prompt = f"""###Instruct: Follow the given review and the 
            corresponded output try to brainstorm and predict the aspect
            and polarity of that aspect in the review 
            ###Input: {review} 
            ###Output: aspect: {aspect} polarity: {polarity} opinion: {opinion} 
            """
        batch["prompt"] = prompt

        return batch

    @staticmethod
    def _create_hf_ds(csv_path: str):
        train_csv = pd.read_csv(csv_path)
        ds = HFDataset.from_pandas(train_csv)
        return ds

    def show_metadata(self):
        for i, row in enumerate(self.dataset):
            review = row["review"]
            aspect = row["aspect"]
            opinion = row["opinion"]
            polarity = row["polarity"]
            prompt = row["prompt"]
            result = {review, prompt, aspect, opinion, polarity}
            print(f"Point: {i} dict: {result}")

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    conf = get_configs("../configs/instruction_tuning_absa.yaml")
    # conf["model"]["pretrained"]["name"] = "google-bert/bert-base-multilingual-cased"

    ds = ABSADataset(conf=conf, csv_path="metadata/generated-manifest.csv")

    ds.show_metadata()
