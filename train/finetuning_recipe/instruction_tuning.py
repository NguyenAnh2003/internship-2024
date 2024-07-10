from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
import torch
import os
from omegaconf import OmegaConf, DictConfig
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    # get_peft_config,
    # get_peft_model_state_dict,
    get_peft_model,
)
from libs.helper_functions import get_configs
from data_manipulation.dataloader import ABSADataset
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


class InstructionTuningLLM:
    def __init__(self, conf: DictConfig = None):
        self.conf = OmegaConf.create(conf)
        self.peft_config = self._setup_peft_config()
        self.bnb_conf = self._setup_4bit_quant_config()  # setup bit and byte config
        self.model, self.tokenizer = self._init_pretrained_llm()

    def _setup_4bit_quant_config(self):
        # self.conf.model.bitandbytes_config.bnb_4bit_compute_dtype = (
        #     torch.float16
        # )  # float 16 or bfloat 16

        config = BitsAndBytesConfig(
            load_in_4bit=self.conf.model.load_in_4bit,  # config load 4 bit
            bnb_4bit_quant_type=self.conf.model.bitandbytes_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=self.conf.model.bitandbytes_config.bnb_4bit_use_double_quant,
        )

        return config

    def _setup_peft_config(self):
        peft_config = LoraConfig(
            lora_alpha=self.conf.model.peft_config.alpha,
            lora_dropout=self.conf.model.peft_config.lora_dropout,
            r=self.conf.model.peft_config.peft_r,
            bias=self.conf.model.peft_config.peft_bias,
            task_type=self.conf.model.peft_config.task_type,
            inference_mode=False,
        )

        return peft_config

    def setup_peft_model(self):
        # peft_config inited in constructor
        peft_model = get_peft_model(self.model, self.peft_config)
        return peft_model

    def _init_pretrained_llm(self):
        model = None #
        tokenizer = AutoTokenizer.from_pretrained(
            self.conf.model.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )  # tokenizer

        # if tokenizer.pad_token is None:
        # tokenizer.add_special_token({'pad_token': '[PAD]'})
        tokenizer.pad_token = tokenizer.eos_token  # replace pad with eos token
        # tokenizer.add_eos_token = True

        # config use_cache: False -> don't use old params
        if self.bnb_conf:
            model = AutoModelForCausalLM.from_pretrained(
                self.conf.model.model_name,
                use_cache=self.conf.model.use_cache,
                torch_dtype=torch.bfloat16,
                quantization_config=self.bnb_conf,  # inited on constructor
                trust_remote_code=True,
            )

        model.config.use_cache = False  # avoid caching params
        model.gradient_checkpointing_enable()  # enable grad check point for not memorize the length chain
        model = prepare_model_for_kbit_training(model)

        return model, tokenizer

    def setup_train_dataset(self, path):
        dataset = ABSADataset(self.tokenizer, self.conf, path)
        dataset.show_metadata() # showing metadata first

    def instruction_tuning(self):
        train_args = TrainingArguments()
        trainer = Trainer()
        trainer.train()  # train !!!

    def inference(self, data_point):
        # prompt template
        def _instruction_template4inference(point):
            return f""" Follow the given review dont summarize, just analyze 
        and predict the corresponded aspect and polarity of train service 
        (the aspect and corresponded polarity must be belongs to train service) 
        ###Input: {point["review"]}
        """

        # calling model
        prompt = _instruction_template4inference(data_point)  # get prompt

        encoding = self.tokenizer(prompt, return_tensors="pt").to("")

        with torch.inference_mode():
            prediction = self.model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                # generation_config=self.conf
            )

        output = self.tokenizer.decode(prediction[0], skip_special_tokens=True)
        return output

if __name__ == "__main__":
    conf = get_configs("../../configs/instruction_tuning_absa.yaml")
    conf["model"]["model_name"] = "google-bert/bert-base-multilingual-cased"
    print(conf)
    iabsa = InstructionTuningLLM(conf=conf)
    # iabsa.setup_train_dataset(path="../../data_manipulation/metadata/generated-manifest.csv")