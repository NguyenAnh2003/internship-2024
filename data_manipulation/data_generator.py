import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
from omegaconf import DictConfig, OmegaConf

load_dotenv()


class Generator:
    def __init__(self, conf: DictConfig = None) -> None:
        # config
        self.conf = OmegaConf.create(conf)

        # api key for llm
        api_key = api_key = os.environ["GEMINI_APIKEY"]
        if self.conf.model.llm.type == "gemini":
            # config gemini
            # call with model name
            genai.configure(api_key=api_key)
            safety_setting = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                },
            ]
            self.llm = genai.GenerativeModel(self.conf.model.llm.name,
                                             safety_settings=safety_setting)

    def llm_task_prediction(self, prompt_template):
        response = self.llm.generate_content(prompt_template)
        return response.text

    def generate_ds(self, out_path: str = None) -> None:
        # out file format is json
        # jsonfile = open(out_path, "a", encoding="utf-8")

        prompt_template = "Give me 100 humanity feedbacks of tourists about Bangkok train service, include tourist's feelings, ending of each example is > sign, the dataset must be natural don't start the sentence identically"
        response = self.llm.generate_content(prompt_template)

        # print(response.text)

        ds = response.text
        # ds = ds.split(">")

        print(ds)

        # for i, item in enumerate(ds):
        #     data_package = {f"feeback {i}": item}
        #     json.dump(data_package, jsonfile, ensure_ascii=False)
        #     jsonfile.write("\n")


if __name__ == "__main__":

    generator = Generator()
    print("DONE")
