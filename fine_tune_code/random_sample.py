import json
import os
import random
import chardet

news_dir = "path to original training sets"
sample_dir = 'path to narrowed set(randomly pick 1000)'
file_list = [f for f in os.listdir(news_dir)]

for file_name in file_list:
    file_path = os.path.join(news_dir, file_name)
    sample_path = os.path.join(sample_dir, file_name)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if len(data) < 1000:
            raise ValueError("Not enough data to sample")
        sample_data = random.sample(data, 1000)
        with open(sample_path, 'w') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=4)
        print(f"saved 1000 sample to {sample_path}")
    except:
        print(f"Failed to sample {sample_path}")