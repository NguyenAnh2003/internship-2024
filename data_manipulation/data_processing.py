from omegaconf import OmegaConf, DictConfig
import re
import json
from google.generativeai import GenerativeModel
from langdetect import detect
import csv

# from data_generator import Generator


class DataProcessPipeline:
    # data processing pipeline with aspect based sentiment analysis
    def __init__(self, conf: DictConfig = None) -> None:
        # input as yaml file
        self.conf = OmegaConf.create(conf)
        # self.llm = GenerativeModel(
        # self.conf.model.llm.name
        # )  # name of llm to preprocess data

        self.emoji_pattern = re.compile(
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

        self.punctuations = [",", ".", "!", "?", ";", ":", "<", ">", "/", "-"]

    def processing_step(self, path: str = None, out_path: str = None):
        result = []
        # json file
        json_file = open(path, "r", encoding="utf-8")
        dataset = json.load(json_file)

        # processing each data point
        for idx, point in enumerate(dataset):
            self.remove_emoji(point["reviewText"])  # remove emoji in text
            self.remove_emoji(point["ratingText"])  # remove emoji in rating text

            # remove html tags
            point["reviewText"] = point["reviewText"].replace("<br />", " ")

            # remove punctuation
            point["reviewText"] = self.remove_punctuation(point["reviewText"])
            # point["reviewTitle"] = self.remove_punctuation(point["reviewTitle"])

            # detect language and group ds
            lang = detect(point["reviewText"])
            if lang == "en":
                result.append(point)

        # write data to out file
        out_file = open(out_path, "w", encoding="utf-8")
        json.dump(result, out_file, indent=4, ensure_ascii=False)

    @staticmethod
    def convert_csv2json(path: str, out_path: str):
        # csv file
        csv_file = open(path, "r", encoding="utf-8")
        # json file
        json_file = open(out_path, "w", encoding="utf-8")
        # csv reader
        reader = csv.reader(csv_file)
        next(reader)  # skip the header

        for line in reader:
            try:
                # define data dict
                data_package = {
                    "id": line[1],
                    "title": line[2],
                    "review": line[3],
                    "rating": line[4],
                    "country": line[5],
                    "polarity": line[6],
                    "month": line[8],
                    "year": line[9],
                }

                # dump file
                json.dump(data_package, json_file, ensure_ascii=False)
                json_file.write("\n")
            except Exception as e:
                raise ValueError(e)

    def remove_punctuation(self, string):
        # Replace specific punctuation characters
        for char in self.punctuations:
            string = string.replace(char, "")
        return string

    def remove_emoji(self, string):
        self.emoji_pattern.sub(r"", string)

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
                if key in point:
                    del point[key]

        outfile = open(out_path, "w", encoding="utf-8")
        json.dump(dataset, outfile, indent=4, ensure_ascii=False)

    @staticmethod
    def get_total_samples(path: str):
        count = 0
        file = open(path, "r", encoding="utf-8")
        ds = json.load(file)

        # counter
        for _ in ds:
            count += 1

        print(count)

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
    # path = "./data_manipulation/metadata/totalData.json"
    # opath = "./data_manipulation/metadata/processed_train.json"

    # remove un cols -> processing_step.
    # pipeline.prepreprocesse_ds(path=path, out_path=opath)
    # pipeline.processing_step(path=opath, out_path=opath)
    # pipeline.get_total_samples(path=opath)

    # convert csv2json
    # csv_path = "./data_manipulation/metadata/total.csv"
    # out_json = "./data_manipulation/metadata/train-manifest.json"
    # pipeline.convert_csv2json(csv_path, out_json)
    
    print("DONE")
