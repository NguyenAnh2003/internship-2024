from omegaconf import OmegaConf, DictConfig
import re
import json
from data_generator import Generator
from langdetect import detect
import csv
import demoji

# from data_generator import Generator

demoji.download_codes()  # download code


class DataProcessPipeline:
    # data processing pipeline with aspect based sentiment analysis
    def __init__(self, conf: DictConfig = None) -> None:
        # input as yaml file
        self.conf = OmegaConf.create(conf)
        self.generator = Generator(self.conf)  # passing conf to Generator class

        self.punctuations = [",", ".", "!", "?", ";", ":", "<", ">", "/", "-"]

    def processing_step(self, path: str = None, out_path: str = None):
        result = []
        # json file
        json_file = open(path, "r", encoding="utf-8")
        dataset = json.load(json_file)

        # processing each data point
        for idx, point in enumerate(dataset):
            self.remove_emoji(point["review"])  # remove emoji in text
            self.remove_emoji(point["rating"])  # remove emoji in rating text

            # remove html tags
            point["review"] = point["review"].replace("<br />", " ")

            # remove punctuation
            point["review"] = self.remove_punctuation(point["review"])
            # point["reviewTitle"] = self.remove_punctuation(point["reviewTitle"])

            # detect language and group ds
            lang = detect(point["review"])
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

        package_list = []

        for line in reader:
            try:
                # define data dict
                data_package = {
                    "id": int(line[1]),
                    "title": line[2],
                    "review": line[3],
                    "rating": int(line[4]),
                    "country": line[5],
                    "polarity": line[6],
                    "month": line[8],
                    "year": line[9],
                    "social": line[-1],
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

    @staticmethod
    def split_dataset(
        path: str = None,
        out_path: str = None,
        train_size: float = 0.7,
        dev_size: float = 0.15,
        test_size: float = 0.15,
    ) -> None:
        assert train_size + dev_size + test_size == 1.0

        # define path
        train_path = out_path + "train-manifest.json"
        dev_path = out_path + "dev-manifest.json"
        test_path = out_path + "test-manifest.json"

        # json file
        dataset = json.load(open(path, "r", encoding="utf-8"))  # read
        train_file = open(train_path, "w", encoding="utf-8")  # write
        dev_file = open(dev_path, "w", encoding="utf-8")  # write
        test_file = open(test_path, "w", encoding="utf-8")  # write

        # dataset len
        length = len(dataset)

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

    def apsect_extraction(self, path: str = None):
        # review
        review = """ I last used the AirRail link back in 2013 bacause 
        it shuts down at midnight It was cheap around B40 to the main 
        station in Makkasan The express line was not running only the 
        city line which stops at every station  Be aware that they 
        sometimes don't go to the end of the line at Phayathai where 
        you can walk to the BTS Skytrain  The station at Makkasan was 
        not wellattended at night and there were no taxis I had to walk
        about 3 city blocks to an intersection to get a cab """

        review2 = """ The train usually late, It's crowded at 
        the peak hour """

        # prompt
        prompt = f"""
        ###Instruct: Follow the given review and extract the aspects 
        that user try to express in the given review, 
        the aspects must be included in the list 
        {self.conf.model.label_aspects} 
        if the aspect not included in given list just define Other
        ###Review: {review2} """

        aspect = self.generator.llm_task_prediction(prompt)

        return aspect


if __name__ == "__main__":
    pipeline = DataProcessPipeline()
    path = "./data_manipulation/metadata/manifests/train-manifest.json"
    path1 = "./data_manipulation/metadata/manifests/temp-ds-metadata.json"
    # opath = "./data_manipulation/metadata/processed_train.json"

    # remove un cols -> processing_step.
    # pipeline.prepreprocesse_ds(path=path, out_path=opath)
    # pipeline.processing_step(path=path, out_path=path)
    # pipeline.get_total_samples(path=path)

    # convert csv2json
    # csv_path = "./data_manipulation/metadata/total.csv"
    # out_json = "./data_manipulation/metadata/manifests/train-manifest.json"
    # pipeline.convert_csv2json(csv_path, out_json)

    # split ds
    # pipeline.split_dataset(
    #     "./data_manipulation/metadata/manifests/train-manifest.json",
    #     "./data_manipulation/metadata/manifests/",
    # )

    print("DONE")
