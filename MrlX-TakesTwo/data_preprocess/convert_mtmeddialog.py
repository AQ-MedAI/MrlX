#!/usr/bin/env python
# coding: utf-8
"""
Convert MTMedDialog parquet data to JSONL format for RL training.

This script processes the MTMedDialog dataset by:
1. Extracting and cleaning content from prompts
2. Applying chat templates using Qwen tokenizer
3. Extracting patient descriptions and solutions
4. Formatting data for downstream RL training
"""

import re
import pandas as pd
from pandas import DataFrame
from transformers import AutoTokenizer


def load_tokenizer(model_name: str = "Qwen/Qwen3-32B") -> AutoTokenizer:
    """Load the tokenizer for chat template application."""
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def extract_user_content(df: DataFrame) -> DataFrame:
    """Extract user content from the prompt field."""
    df["content"] = df["prompt"].apply(lambda x: x[0]["content"])
    df["content"] = df["content"].str.extract(
        r"<\|im_start\|>user\n(.*?)<\|im_end\|>", flags=re.DOTALL
    )[0]
    return df


def update_response_instructions(df: DataFrame) -> DataFrame:
    """Replace response instructions with standardized format."""
    new_response_text = """If you believe there is insufficient information, please only ask one question, in this format:
Question: (your question).
If you believe you have obtained enough information, please only provide diagnosis and recommendations, in this format:
Diagnosis: (the patient's most likely disease or symptoms)
Recommendation: (corresponding treatment plan or advice)"""

    # Replace text between "Response:" and "Rewards:"
    df["content"] = df["content"].str.replace(
        r"Response:\n(.*?)\nRewards:",
        f"Response:\n{new_response_text}\nRewards:",
        flags=re.DOTALL,
        regex=True,
    )

    # Clean up after "Decide next action:"
    df["content"] = df["content"].str.replace(
        r"(Decide next action:).*", r"\1", flags=re.DOTALL, regex=True
    )

    return df


def apply_chat_template(df: DataFrame, tokenizer: AutoTokenizer) -> DataFrame:
    """Apply chat template to messages."""
    df["messages"] = df["content"].apply(lambda x: [{"content": x, "role": "user"}])
    df["prompt"] = df["messages"].apply(
        lambda x: tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            x, add_generation_prompt=True, tokenize=False
        )
    )
    return df


def extract_metadata(df: DataFrame) -> DataFrame:
    """Extract patient description and solution metadata."""
    # Extract patient description
    df["description"] = (
        df["content"]
        .str.extract(
            r"Patient's description:\s*(.*?)(?=\n\nDecide next action:|$)",
            flags=re.DOTALL,
        )[0]
        .str.strip()
    )

    # Extract enhanced description (self report)
    df["self_report"] = df["reward_model"].apply(lambda x: x["enhanced_description"])

    # Extract ground truth solution
    df["solution"] = df["reward_model"].apply(lambda x: x["ground_truth"])

    # Merge into extra_info
    df["extra_info"] = df.apply(
        lambda row: {
            **row["extra_info"],
            "solution": row["solution"],
            "reward_model": "grm",
            "chief_complaint": row["description"],
            "self_report": row["self_report"],
        },
        axis=1,
    )

    return df


def main():
    """Main processing pipeline."""
    print("Loading data...")
    df = pd.read_parquet("MTMedDialog_RL.parquet")
    print(f"Loaded {len(df)} records")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("Extracting user content...")
    df = extract_user_content(df)

    print("Updating response instructions...")
    df = update_response_instructions(df)

    print("Sample content:")
    print(df["content"].iloc[0])
    print("\n" + "=" * 80 + "\n")

    print("Applying chat template...")
    df = apply_chat_template(df, tokenizer)

    print("Sample prompt with chat template:")
    print(df["prompt"].iloc[0])
    print("\n" + "=" * 80 + "\n")

    print("Extracting metadata...")
    df = extract_metadata(df)

    print("Sample extra_info:")
    print(df["extra_info"].iloc[0])
    print("\n" + "=" * 80 + "\n")

    print("Saving to JSONL...")
    output_file = "MTMedDialog_RL.jsonl"
    df[["extra_info", "messages"]].to_json(
        output_file, orient="records", lines=True, force_ascii=False
    )

    print(f"Successfully saved {len(df)} records to {output_file}")


if __name__ == "__main__":
    main()
