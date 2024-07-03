from omegaconf import OmegaConf, DictConfig
import re
import json
import spacy
from google.generativeai import GenerativeModel


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
        rs = json.load(json_file)
        print(rs)

        # out_file = open(out_path, 'w', encoding='utf-8')

    def remove_unnecessary_cols(self, path: str, out_path: str = None):
        json_file = open(path, "r", encoding="utf-8")
        dataset = json.load(json_file)
        for point in dataset:
            print(f"Data point: {point}")

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
    pipeline.remove_unnecessary_cols("./data_manipulation/metadata/train.json")
