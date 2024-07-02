import google.generativeai as genai
import os
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
from transformers import AutoModel

load_dotenv()

LLM_KEY = os.environ.get("GEMINI_APIKEY")


class LLMModule:
    def __init__(self, conf: DictConfig = None) -> None:
        self.conf = OmegaConf.create(conf)

    def prompt_template(self, st):
        return

    def get_llm_model(self):
        model = None # init None model
        if self.conf.model.gemini == True: # checking using gemini?
            model = genai.GenerativeModel(self.conf.model.name)
        else:
            return
        
        return model
            
