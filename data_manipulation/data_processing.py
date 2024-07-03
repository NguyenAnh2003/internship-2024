from omegaconf import OmegaConf, DictConfig
import re
import json
import spacy
from google.generativeai import GenerativeModel
from uuid import uuid4


class DataProcessPipeline:
    # data processing pipeline with aspect based sentiment analysis
    def __init__(self, conf: DictConfig = None) -> None:
        # input as yaml file
        self.conf = OmegaConf.create(conf)
        # self.llm = GenerativeModel(
        # self.conf.model.llm.name
        # )  # name of llm to preprocess data

    def processing_step(self, path: str = None, out_path: str = None):
        """
        flow of processing data js format -> filter with regex (punctuation, number and begining)
        -> trim text
        """
        # json file
        json_file = open(path, "r", encoding="utf-8")
        dataset = json.load(json_file)

        for point in dataset:
            self.remove_emoji(point["reviewText"])
            self.remove_emoji(point["ratingText"])

        out_file = open(out_path, "w", encoding="utf-8")
        json.dump(dataset, out_file, indent=4, ensure_ascii=False)

    def remove_emoji(self, string):
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002500-\U00002BEF"  # chinese char
            "\U00002702-\U000027B0"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\u2022"
            "\ufe0f"  # dingbats
            "\u3030"
            "]+",
            flags=re.UNICODE,
        )

        emoji_pattern.sub(r"", string)

    def prepreprocesse_ds(self, path: str, out_path: str = None):
        json_file = open(path, "r", encoding="utf-8")
        dataset = json.load(json_file)

        keys = [
            "userName",
            "userProfile",
            "userAvatar",
            "tripType",
            "helpfulVotes",
            "photos",
            "publishedDate",
            "disclaimer",
        ]

        for key in keys:
            for point in dataset:
                point["id"] = str(uuid4())  # adding id to each sample
                if key in point:
                    del point[key]

        outfile = open(out_path, "w", encoding="utf-8")
        json.dump(dataset, outfile, indent=4, ensure_ascii=False)

    def regex_filter(self, sample: str = None) -> None:
        pattern = r"\n([0-9][0-9][0-9])\."  # pattern to filter

        str0 = re.sub(
            pattern,
            "",
            "\n100. The train was a bit slow, but it was still a comfortable way to get around the city.",
        )

        print(str0.strip())

    def apsect_extraction(self, sample: str = None):
        aspect = ""
        return aspect


if __name__ == "__main__":
    pipeline = DataProcessPipeline()
    path = "./data_manipulation/metadata/train.json"
    opath = "./data_manipulation/metadata/processed_train.json"
    pipeline.processing_step(path=opath, out_path=path)
    print("DONE")
