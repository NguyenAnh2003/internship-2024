import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_APIKEY"])

model = genai.GenerativeModel("gemini-1.5-flash")

class Generator:
    def __init__(self) -> None:
        pass
    
    def prompt_template(self):
        pass
    
    def generate_ds(self):
        pass


def prompt_template(example: str) -> str:
    
    return


def gen_ds():
    jsonfile = open("./data_manipulation/metadata/gen_ds.json", "a", encoding="utf-8")

    prompt_template = "Give me 100 humanity feedbacks of tourists about Bangkok train service, include tourist's feelings, ending of each example is > sign, the dataset must be natural don't start the sentence identically"
    response = model.generate_content(prompt_template)

    # print(response.text)

    ds = response.text
    ds = ds.split(">")

    for i, item in enumerate(ds):
        data_package = {f"feeback {i}": item}
        json.dump(data_package, jsonfile, ensure_ascii=False)
        jsonfile.write("\n")


for _ in range(100):
    gen_ds()

print("DONE")
