import os
import random
import pandas as pd

DATASET_DIR = "data/common_voice/"

def load_sentences_and_paths(tsv_file):
    """加载句子和对应的音频路径"""
    df = pd.read_csv(tsv_file, sep='\t', usecols=["path", "sentence"])
    return df.to_dict(orient="records")

def filter_invalid_data(data_records, invalidated_file):
    """过滤无效的录音"""
    if not os.path.exists(invalidated_file):
        return data_records
    with open(invalidated_file, "r", encoding="utf-8") as f:
        invalid_paths = set(line.strip().split("\t")[0] for line in f)
    return [record for record in data_records if record["path"] not in invalid_paths]

def get_random_sentence(filtered_data):
    """随机选择一句句子"""
    return random.choice(filtered_data)

