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


if __name__ == "__main__":

    generator = Generator()
    print("DONE")
