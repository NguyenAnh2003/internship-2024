from data_processing import DataProcessPipeline
from libs.helper_functions import get_configs
import time

if __name__ == "__main__":
    conf = get_configs("../configs/absa_model.yaml")
    conf["model"]["llm"]["type"] = "gemini"
    conf["model"]["llm"]["name"] = "gemini-1.5-flash"
    pipeline = DataProcessPipeline(conf)

    # path = "metadata/manifests/train-manifest1.1.1.1.1.1.json"
    path = "metadata/manifests/dev-clean-manifest.json"

    # pipeline.apsect_extraction(
    #     path=path,
    #     out_path=f"./metadata/manifests/temp-manifest{time.time()}.json"
    # )

    pipeline.get_total_samples(path)
