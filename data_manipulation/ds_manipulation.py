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
    path = "metadata/manifests/train-1-manifest.json"
    path1 = "metadata/manifests/train-clean-manifest.json"
    file = open(path, 'r', encoding='utf-8')
    wfile = open(path1, 'w', encoding='utf-8')
    ds = json.load(file)
    rs = []

    # for idx, point in enumerate(ds):
    #     # d = point["labels"]["aspect"]
    #     # id = point["id"]
    #     # print(f"{d} {id}")
    #     data = {**point, "aspect": point["labels"]["aspect"],
    #             "opinion": point["labels"]["opinion"],
    #             "polarity": point["labels"]["polarity"]}
    #     del point["labels"]
    #     rs.append(data)

    for idx, point in enumerate(ds):
        del point["labels"]
        point["polarity"] = point["polarity"].lower()
        rs.append(point)

    json.dump(rs, wfile, ensure_ascii=False, indent=4)


    # pipeline.apsect_extraction(
    #     path=path,
    #     out_path=f"./metadata/manifests/temp/temp-manifest{time.time()}.json"
    # )

    # pipeline.get_total_samples(path)
