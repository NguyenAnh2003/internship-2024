from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datasets import Dataset as HFDataset
from omegaconf import OmegaConf, DictConfig


class ABSADataset(Dataset):
    def __init__(self, tokenizer, csv_path, conf: DictConfig = None) -> None:
        super().__init__()
        self.conf = OmegaConf.create(conf)
        self.tokenizer = tokenizer  # tokenizer as transform
        self.dataset = self._create_hf_ds(csv_path=csv_path)  # create HF dataset

    def __getitem__(self, index):
        prompt, review, aspect, opinion, polarity = self._instruction_template4train(
            idx=index
        )
        return {prompt, review, aspect, opinion, polarity}

    def _instruction_template4train(self, index):
        review = self.dataset[index]["review"]  # tokenize this
        aspect = self.dataset[index]["aspect"]
        opinion = self.dataset[index]["opinion"]
        polarity = self.dataset[index]["polarity"]
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
