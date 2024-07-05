from data_processing import DataProcessPipeline
from libs.helper_functions import get_configs

if __name__ == "__main__":
    conf = get_configs("../configs/absa_model.yaml")
    conf["model"]["llm"]["type"] = "gemini"
    conf["model"]["llm"]["name"] = "gemini-1.5-flash"
    print(conf)
    pipeline = DataProcessPipeline(conf)

    aspect = pipeline.apsect_extraction()
    print(aspect)