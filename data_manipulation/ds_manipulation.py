from data_processing import DataProcessPipeline
from libs.helper_functions import get_configs
import json
import time

if __name__ == "__main__":
    conf = get_configs("../configs/absa_model.yaml")
    conf["model"]["llm"]["type"] = "gemini"
    conf["model"]["llm"]["name"] = "gemini-1.5-flash"
    pipeline = DataProcessPipeline(conf)

    # path = "metadata/manifests/train-manifest1.1.1.1.1.1.json"
    path1 = "metadata/manifests/train-1-manifest.json"
    path = "metadata/manifests/train-clean-manifest.json"
    file = open(path, 'r', encoding='utf-8')
    wfile = open(path1, 'w', encoding='utf-8')
    ds = json.load(file)
    rs = []

    label_aspects = ["Safety", "Cleanliness", "Data availability", "Price fairness",
                    "User-friendly payment system", "Satisfactions", "Facilities at station",
                    "Facilities within train", "Accessibility for disabilities", "Maintenance",
                    "Punctuality", "Ease of connections between lines", "Reliability equipment",
                    "Accidents", "Others"]

    # for idx, point in enumerate(ds):
    #     if point["aspect"] not in label_aspects:
    #         del point
    #     else:
    #         rs.append(point)

    # json.dump(rs, wfile, ensure_ascii=False, indent=4)


    # pipeline.apsect_extraction(
    #     path=path,
    #     out_path=f"./metadata/manifests/temp/temp-manifest{time.time()}.json"
    # )

    pipeline.get_total_samples(path)