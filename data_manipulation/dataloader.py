from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datasets import Dataset as HFDataset
from omegaconf import OmegaConf, DictConfig
from libs.helper_functions import get_configs


class ABSADataset(Dataset):
    def __init__(self, tokenizer = None, csv_path = "", conf: DictConfig = None) -> None:
        super().__init__()
        self.conf = OmegaConf.create(conf)
        self.tokenizer = tokenizer  # tokenizer as transform
        self.dataset = self._create_hf_ds(csv_path=csv_path)  # create HF dataset

    def __getitem__(self, index):
        pass

    def _instruction_template4train(self, index):
        review = self.dataset[index]["review"]  # tokenize this
        aspect = self.dataset[index]["aspect"] # label2index
        opinion = self.dataset[index]["opinion"] # tokenize this
        polarity = self.dataset[index]["polarity"] # label2index
        prompt = ""
        # take review as input and aspect, polarity as output
        if self.conf.model.instruction_style == "simple":
            prompt = f""" 
            ###Instruct: Follow the given review and the corresponded output
            try to brainstorm and predict the aspect and polarity of that aspect in the review
            ###Input: {review}
            ###Output: aspect: {aspect} polarity: {polarity}
            """

        return prompt, review, aspect, opinion, polarity

    @staticmethod
    def _create_hf_ds(csv_path: str):
        train_csv = pd.read_csv(csv_path)
        ds = HFDataset.from_pandas(train_csv)
        return ds

    def __len__(self):
        return len(self.dataset)


class ABSAloader(DataLoader):
    def __init__(self) -> None:
        super.__init__()

if __name__ == "__main__":
    conf = get_configs("../configs/instruction_tuning_absa.yaml")
    ds = ABSADataset(conf,
                     csv_path="../data_manipulation/metadata/generated-manifest.csv")
    print(ds.dataset)