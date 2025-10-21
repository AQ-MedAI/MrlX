# Copyright 2025 Ant Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate mock JSONL files for sub-agent data preparation."""

import argparse
import json
import logging
import os
import sys


logger = logging.getLogger(__name__)


def create_mock_file(file_path: str, n: int):
    """Create a mock JSONL file based on input file structure.

    Args:
        file_path: Path to input JSONL file.
        n: Line count multiplier.
    """
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        sys.exit(1)

    logger.info(f"Processing input file: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if not first_line:
                logger.error("Input file is empty")
                sys.exit(1)

        first_record_dict = json.loads(first_line)
        logger.info(f"Successfully parsed first line with {len(first_record_dict)} keys")

        mock_record_dict = {key: "" for key in first_record_dict.keys()}
        mock_record_str = json.dumps(mock_record_dict, ensure_ascii=False)

        line_count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                line_count += 1

        logger.info(f"Original file line count: {line_count}")

        total_new_lines = line_count * n
        logger.info(f"Generating new file with {line_count} * {n} = {total_new_lines} lines")

        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        output_path = os.path.join(dir_name, f"mock4sub_{base_name}")

        logger.info(f"Writing to: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(total_new_lines):
                f.write(mock_record_str + '\n')

        logger.info("Task completed successfully")

    except json.JSONDecodeError:
        logger.error(f"First line is not valid JSON: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate mock submission file for JSONL data.")
    parser.add_argument(
        "-f", "--file_path",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )

    parser.add_argument(
        "-n", "--n",
        type=int,
        required=True,
        help="Line count multiplier"
    )

    args = parser.parse_args()

    create_mock_file(args.file_path, args.n)
