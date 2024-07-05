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

from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


class InstructionTuningLLM:
    def __init__(self, conf: DictConfig = None) -> None:
        self.conf = OmegaConf.create(conf)
        self.model, self.tokenizer = self._init_pretrained_llm()
        self.peft_config = self._setup_peft_config()
        self.bnb_config = self._setup_4bit_quant_config()  # setup bit and byte config

    def _setup_4bit_quant_config(self):
        self.conf.model.bitandbytes_config.bnb_4bit_compute_dtype = (
            torch.float16
        )  # float 16 or bfloat 16

        config = BitsAndBytesConfig(
            load_in_4bit=self.conf.model.common.load_in_4bit,  # common config load 4 bit
            bnb_4bit_quant_type=self.conf.model.bitandbytes_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.conf.model.bitandbytes_config.bnb_4bit_compute_dtype,
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

        tokenizer = AutoTokenizer.from_pretrained(
            self.conf.model.model_name,
            trust_remote_code=True,
            torch_dtype=self.conf.model.common.torch_dtype,
        )  # tokenizer

        # if tokenizer.pad_token is None:
        # tokenizer.add_special_token({'pad_token': '[PAD]'})
        tokenizer.pad_token = tokenizer.eos_token  # replace pad with eos token
        # tokenizer.add_eos_token = True

        # config use_cache: False -> don't use old params
        model = AutoModelForCausalLM.from_pretrained(
            self.conf.model.model_name,
            use_cache=self.conf.model.common.use_cache,
            torch_dtype=self.conf.model.common.torch_dtype,
            load_in_4bit=self.conf.model.common.load_in_4bit,
            quantization_config=self.bnb_config,  # inited on constructor
            trust_remote_code=True,
        )

        """ getting model for kbit quantization
		Casts all the non kbit modules to full precision(fp32) for stability
		Adds a forward hook to the input embedding layer to calculate the
		gradients of the input hidden states
		Enables gradient checkpointing for more memory-efficient training
		"""

        model.config.use_cache = False  # avoid caching params
        model.gradient_checkpointing_enable()  # enable grad check point for not memorize the length chain
        model = prepare_model_for_kbit_training(model)

        return model, tokenizer

    def _instruction_template(self, data_point):
        return f""

    def instruction_tuning():
        train_args = TrainingArguments()
        trainer = Trainer()
        trainer.train()  # train !!!
