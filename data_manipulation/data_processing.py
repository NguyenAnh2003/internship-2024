from omegaconf import OmegaConf, DictConfig
import re


class DataProcessPipeline:
    # data processing pipeline with aspect based sentiment analysis
    def __init__(self, conf: DictConfig = None) -> None:
        # input as yaml file
        self.conf = OmegaConf.create(conf)

    def processing_step(self, path: str = None, out_path: str = None):
        """
        flow of processing data js format -> filter with regex (punctuation, number and begining)
        -> trim text
        """
        # json file
        json_file = open(path, "r", encoding="utf-8")

        # out_file = open(out_path, 'w', encoding='utf-8')

    def regex_filter(self, sample: str = None) -> None:
        pattern = r"\n([0-9][0-9][0-9])\."  # pattern to filter

        str0 = re.sub(
            pattern,
            "",
            "\n100. The train was a bit slow, but it was still a comfortable way to get around the city.",
        )

        print(str0.strip())

    def apsect_extraction(self, sample: str = None):
        aspect = ""
        return aspect


if __name__ == "__main__":
    pipeline = DataProcessPipeline()
    pipeline.processing_step("./data_manipulation/metadata/gen_ds.json")
