import pandas as pd

full_data = pd.read_json("./data/2025-05-07-06-14-12_oss_eval.jsonl", lines=True)
hard_data = pd.read_json("./data/hard_2025-05-08-21-00-10.jsonl", lines=True)

full_data["is_hard"] = full_data["prompt_id"].isin(hard_data["prompt_id"])

full_data[full_data["is_hard"]].to_json("./data/healthbench_easy.jsonl", orient="records", lines=True)
full_data[~full_data["is_hard"]].to_json("./data/healthbench_hard.jsonl", orient="records", lines=True)

print(full_data.head())