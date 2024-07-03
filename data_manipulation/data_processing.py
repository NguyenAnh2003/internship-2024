from omegaconf import OmegaConf, DictConfig
import re
import json
import spacy
from google.generativeai import GenerativeModel
from uuid import uuid4
from langdetect import detect


class DataProcessPipeline:
    # data processing pipeline with aspect based sentiment analysis
    def __init__(self, conf: DictConfig = None) -> None:
        # input as yaml file
        self.conf = OmegaConf.create(conf)
        # self.llm = GenerativeModel(
        # self.conf.model.llm.name
        # )  # name of llm to preprocess data

    def processing_step(self, path: str = None, out_path: str = None):
        result = []
        # json file
        json_file = open(path, "r", encoding="utf-8")
        dataset = json.load(json_file)

        # processing each data point
        for idx, point in enumerate(dataset):
            self.remove_emoji(point["reviewText"])  # remove emoji in text
            self.remove_emoji(point["ratingText"])  # remove emoji in rating text
            lang = detect(point["reviewText"])
            if lang == "en":
                result.append(point)

        out_file = open(out_path, "w", encoding="utf-8")
        json.dump(result, out_file, indent=4, ensure_ascii=False)

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
        pass

    def check_ds_lang(self, path):
        json_file = open(path, "r", encoding="utf-8")
        dataset = json.load(json_file)
        
        for point in dataset:
            lang = detect(point["reviewText"])
            print(lang)

    def apsect_extraction(self, sample: str = None):
        aspect = ""
        return aspect


if __name__ == "__main__":
    pipeline = DataProcessPipeline()
    path = "./data_manipulation/metadata/train.json"
    opath = "./data_manipulation/metadata/processed_train.json"
    pipeline.processing_step(path=path, out_path=opath)
    print("DONE")
