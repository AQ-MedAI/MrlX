# Self-Rewarding for Open-Ended Generation

## Data Preprocessing

Download the evaluation datasets and convert to train/test splits.

```bash
# Create data directory
mkdir -p ./data

# Download standard evaluation dataset
curl -o ./data/2025-05-07-06-14-12_oss_eval.jsonl "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"

# Download hard evaluation dataset
curl -o ./data/hard_2025-05-08-21-00-10.jsonl "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"

# Convert to train/test splits
python data_preprocess.py
```

