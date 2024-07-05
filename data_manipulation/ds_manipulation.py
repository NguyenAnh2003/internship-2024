from data_generator import Generator
from libs.helper_functions import get_configs


if __name__ == "__main__":
    conf = get_configs("../configs/instruction_tuning_absa.yaml")
    generator = Generator(conf=conf)
    generator.generate_ds()