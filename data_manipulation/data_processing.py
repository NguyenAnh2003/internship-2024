import time

from omegaconf import OmegaConf, DictConfig
import re
import json
from data_generator import Generator
from langdetect import detect
import csv
import demoji
import random
import os
import ast
from libs.helper_functions import get_configs

# from data_generator import Generator

demoji.download_codes()  # download code


class DataProcessPipeline:
    # data processing pipeline with aspect based sentiment analysis
    def __init__(self, conf: DictConfig = None) -> None:
        # input as yaml file
        self.conf = OmegaConf.create(conf)
        self.generator = Generator(self.conf)  # passing conf to Generator class

        self.punctuations = [",", ".", "!", "?", ";", ":", "<", ">", "/", "-"]

    def processing_step(self, path: str = None, outpath: str = None):
        result = []
        # json file
        json_file = open(path, "r", encoding="utf-8")
        dataset = json.load(json_file)

        # processing each data point
        for idx, point in enumerate(dataset):
            self.remove_emoji(point["review"])  # remove emoji in text
            self.remove_emoji(point["rating"])  # remove emoji in rating text

            # remove html tags
            if "<br />" in point["review"]:
                point["review"] = point["review"].replace("<br />", " ")

            # remove punctuation
            point["review"] = self.remove_punctuation(point["review"])
            # point["reviewTitle"] = self.remove_punctuation(point["reviewTitle"])

            # detect language and group ds
            lang = detect(point["review"])
            if lang == "en":
                result.append(point)

        # write data to out file
        out_file = open(outpath, "w", encoding="utf-8")
        json.dump(result, out_file, indent=4, ensure_ascii=False)

    @staticmethod
    def convert_json2csv(path: str = None, outpath: str = None) -> None:
        json_ds = json.load(open(path, 'r', encoding='utf-8'))
        csv_file = open(outpath, mode='w', encoding='utf-8', newline='')
        writer = csv.writer(csv_file)

        writer.writerow(json_ds[0].keys()) # write header
        
        for idx, point in enumerate(json_ds):
            writer.writerow(point.values())
        

    @staticmethod
    def convert_csv2json(path: str, outpath: str):
        # csv file
        csv_file = open(path, "r", encoding="utf-8")
        # json file
        json_file = open(outpath, "w", encoding="utf-8")
        # csv reader
        reader = csv.reader(csv_file)
        next(reader)  # skip the header

        package_list = []

        for line in reader:
            try:
                # define data dict
                data_package = {
                    "title": line[0],
                    "review": line[-3],
                    "country": line[-2],
                    "aspect": line[4],
                    "opinion": line[5],
                    "polarity": line[-1],
                    "month": line[1],
                    "year": int(line[2]),
                    "social": line[3],
                }

                package_list.append(data_package)
            except Exception as e:
                raise ValueError(e)

        json.dump(package_list, json_file, ensure_ascii=False)
        json_file.write("\n")
        package_list.clear()  # clear list to remove from memory

    def remove_punctuation(self, string):
        # Replace specific punctuation characters
        for char in self.punctuations:
            string = string.replace(char, "")
        return string

    def remove_emoji(self, string):
        # self.emoji_pattern.sub(r"", string)
        dem = demoji.findall(string)
        for item in dem.keys():
            string = string.replace(item, "")
        return string

    def prepreprocesse_ds(self, path: str, outpath: str = None):
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

        outfile = open(outpath, "w", encoding="utf-8")
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

    @staticmethod
    def split_dataset(
        path: str = None,
        outpath: str = None,
        train_size: float = 0.7,
        dev_size: float = 0.15,
        test_size: float = 0.15,
    ) -> None:
        assert train_size + dev_size + test_size == 1.0

        # define path
        # train_path = outpath + "train-manifest1.3.1.json"
        train_path = outpath + ".1.json"
        # dev_path = outpath + "train-manifest1.3.2.json"
        dev_path = outpath + ".2.json"
        # test_path = outpath + "train-manifest1.3.3.json"
        test_path = outpath + ".3.json"

        # json file
        dataset = json.load(open(path, "r", encoding="utf-8"))  # read
        train_file = open(train_path, "w", encoding="utf-8")  # write
        dev_file = open(dev_path, "w", encoding="utf-8")  # write
        test_file = open(test_path, "w", encoding="utf-8")  # write

        # dataset len
        length = len(dataset)
        random.shuffle(dataset)  # shuffle the ds

        train_len = int(train_size * length)
        dev_len = int((train_size + dev_size) * length)

        train_ds = dataset[:train_len]
        dev_ds = dataset[train_len:dev_len]
        test_ds = dataset[dev_len:]

        json.dump(train_ds, train_file, ensure_ascii=False, indent=4)
        json.dump(dev_ds, dev_file, ensure_ascii=False, indent=4)
        json.dump(test_ds, test_file, ensure_ascii=False, indent=4)

    def check_ds_lang(self, path):
        json_file = open(path, "r", encoding="utf-8")
        dataset = json.load(json_file)

        for point in dataset:
            lang = detect(point["reviewText"])
            print(lang)

    def apsect_extraction(self, path: str = None, outpath: str = None):
        dataset = json.load(open(path, "r", encoding="utf-8"))
        outfile = open(outpath, "a", encoding="utf-8")
        result = []  # temporary results
        count = 0

        for idx, point in enumerate(dataset[650:1000]):
            review = point["review"]

            # prompt
            prompt = f"""
            ###Instruct: Follow the given review analyze the review 
            in detail and extract the aspects that user try to express in 
            the given review, the aspects must be included in the list 
            {self.conf.model.label_aspects} if the aspect not included 
            in given list just define Other, if review does not mention
            mention about any aspect in the list don't give any result
            Result template must be in json format: 
            aspect: opinion, polarity (must have aspect and opinion and polarity)
            ###Review: {review} """

            aspect = self.generator.llm_task_prediction(prompt)
            aspect = aspect.replace("```", "")  # remove str
            aspect = aspect.replace("json", "")  # remove str
            point["labels"] = aspect  # convert 2 list
            count += 1
            print(f"count: {count} aspect: {aspect}")
            result.append(point)
            time.sleep(4)

        json.dump(result, outfile, ensure_ascii=False, indent=4)
        result.clear()  # clear after added

    # bind json file
    @staticmethod
    def bind_jsonfile(outpath: str = None) -> None:
        outfile = open(outpath, "a", encoding="utf-8")

        dir = "./metadata/manifests/pp/"
        package_list = []

        for item in os.listdir(dir):
            filepath = os.path.join(dir, item)
            if filepath.endswith(".json"):
                infile = open(filepath, "r", encoding="utf-8")
                dataset = json.load(infile)
                for idx, point in enumerate(dataset):
                    package_list.append(point)

        json.dump(package_list, outfile, ensure_ascii=False, indent=4)
        package_list.clear()

    # preprocessing after labeling
    @staticmethod
    def preprocessing_af_labeling(path: str, outpath: str):
        infile = open(path, "r", encoding="utf-8")
        outfile = open(outpath, "w", encoding="utf-8")

        dataset = json.load(infile)
        rs = []
        for point in dataset:
            len_labels = len(point["labels"])
            if len_labels == 1:
                data = {**point, "labels": point["labels"][0]}
                rs.append(data)
            if len_labels > 1:
                for item in point["labels"]:
                    data = {**point, "labels": item}
                    rs.append(data)

        json.dump(rs, outfile, ensure_ascii=False, indent=4)
        rs.clear()

    @staticmethod
    def process_ate_dataset(path, outpath):
        # raw dataset must have (title, country, opinion, polarity, month, year, social)
        # these needed to be removed
        file = open(path, 'r', encoding="utf-8")
        outfile = open(outpath, 'w', encoding='utf-8')
        ds = json.load(file)
        rs = []
        for point in ds:
            del point["title"]
            del point["country"]
            del point["opinion"]
            del point["polarity"]
            del point["month"]
            del point["year"]
            del point["social"]
            rs.append(point)
        json.dump(rs, outfile, ensure_ascii=False, indent=4)
        rs.clear()



if __name__ == "__main__":
    conf = get_configs("../configs/absa_model.yaml")
    conf["model"]["llm"]["type"] = "gemini"
    conf["model"]["llm"]["name"] = "gemini-1.5-flash"

    pipeline = DataProcessPipeline(conf)
    outpath = "metadata/manifests/ate/ate-processed-manifest.csv"
    path = "metadata/manifests/ate/ate-processed-1-manifest.json"
    pipeline.convert_json2csv(path, outpath)
    print("DONE")
