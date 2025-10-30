# Self-Rewarding for Open-Ended Generation

## Prerequisite

Use the latest slime docker `slimerl/slime:latest`. Assume your slime dir is in `$SLIME_DIR`

```bash
cd $SLIME_DIR
pip install -e .
```

Soft link `MrlX-SelfRewarding` dir into `slime/examples`

```bash
ln -s "<path-to-MrlX-SelfRewarding>" $SLIME_DIR/examples/MrlX-SelfRewarding
```

## Data Preprocessing

Download the evaluation datasets and convert to train/test splits.

```bash
# Create data directory
mkdir -p ./data

# Download standard evaluation dataset
wget -O ./data/2025-05-07-06-14-12_oss_eval.jsonl "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"

# Download hard evaluation dataset
wget -O ./data/hard_2025-05-08-21-00-10.jsonl "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"

# Convert to train/test splits
python data_preprocess.py
```

