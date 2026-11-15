import os


class DefaultDataModuleConfig:
    DATASETS_SAVE_PATH = "./datasets/downloads/"
    NUM_WORKERS = int(os.cpu_count() / 2)
    BATCH_SIZE = 16
