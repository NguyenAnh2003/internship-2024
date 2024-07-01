from omegaconf import OmegaConf, DictConfig
import re


class DataProcessPipeline:
    # data processing pipeline with aspect based sentiment analysis
    def __init__(self, conf: DictConfig = None) -> None:
        # input as yaml file
        self.conf = OmegaConf.create(conf)

    def processing_step(self, path=None, out_path=None):
        # json file
        json_file = open(path, "r", encoding="utf-8")

        pattern = r"\n([0-9][0-9][0-9])\."
        
        str0 = re.sub(
            pattern,
            "",
            "\n100. The train was a bit slow, but it was still a comfortable way to get around the city.",
        )

        # out_file = open(out_path, 'w', encoding='utf-8')


if __name__ == "__main__":
    pipeline = DataProcessPipeline()
    pipeline.processing_step("./data_manipulation/metadata/gen_ds.json")
