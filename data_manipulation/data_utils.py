import demoji
from langdetect import detect
import csv
import json

demoji.download_codes()  # download code

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
                "id": int(line[0]),
                "title": line[1],
                "review": line[2],
                "country": line[3],
                "aspect": line[4],
                "sentiment": line[5],
                "opinion": line[6],
                "month": line[7],
                "year": int(line[8]),
                "social": line[9],
                "original_text": line[-1]
            }

            package_list.append(data_package)
        except Exception as e:
            raise ValueError(e)

    json.dump(package_list, json_file, ensure_ascii=False)
    # json_file.write("\n")
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

def check_ds_lang(self, path):
    json_file = open(path, "r", encoding="utf-8")
    dataset = json.load(json_file)

    for point in dataset:
        lang = detect(point["reviewText"])
        print(lang)