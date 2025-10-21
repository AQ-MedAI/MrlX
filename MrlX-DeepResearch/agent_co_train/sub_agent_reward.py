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

"""Sub-agent reward computation with format validation."""

import re
from typing import Any, Dict, List, Tuple


def _validate_assistant_content(content: str) -> Tuple[bool, str]:
    """Validate single assistant message content format.

    Args:
        content: Assistant message content string.

    Returns:
        Tuple of (is_valid, error_message).
    """
    for tag in ["thinking", "tool_call", "result"]:
        opening_count = content.count(f"<{tag}>")
        closing_count = content.count(f"</{tag}>")
        if opening_count != closing_count:
            return False, f"Tag Mismatch: Found {opening_count} <{tag}> but {closing_count} </{tag}>."

    split_pattern = r"(<(?:thinking|tool_call|result)>|</(?:thinking|tool_call|result)>)"
    parts = re.split(split_pattern, content)

    state = "start"
    for part in parts:
        if not part or not part.strip():
            continue

        is_tag = re.match(r"</?(?:thinking|tool_call|result)>", part)

        if is_tag:
            if part == "<thinking>" and state == "start":
                state = "in_thinking"
            elif part == "</thinking>" and state == "in_thinking":
                state = "after_thinking"
            elif part == "<tool_call>" and state in ["start", "after_thinking"]:
                state = "in_tool_call"
            elif part == "</tool_call>" and state == "in_tool_call":
                state = "after_tool_call"
            elif part == "<result>" and state == "after_thinking":
                state = "in_result"
            elif part == "</result>" and state == "in_result":
                state = "end"
            else:
                return False, f"Invalid tag sequence: Found tag '{part}' in state '{state}'."
        else:
            if state not in ["in_thinking", "in_tool_call", "in_result"]:
                return False, f"Unexpected text found between tags: '{part.strip()[:50]}...'"

    if state not in ["after_tool_call", "end"]:
        return False, f"Turn ended in an invalid state: '{state}'."

    return True, "Valid"


def compute_format_score(messages: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
    """Validate message list format with binary scoring.

    Returns 0 for any format error, 1 if all formats are correct.

    Args:
        messages: List of message dictionaries.

    Returns:
        Tuple of (score, reason_list).
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return 0.0, ["Input is not a list or is too short (less than 2 messages)."]

    if len(messages) < 3:
        return 1.0, ["No agent trajectory to score (messages list has exactly 2 elements)."]

    trajectory = messages[2:]

    expected_role = "assistant"

    for i, msg in enumerate(trajectory):
        current_role = msg.get("role")
        content = msg.get("content", "")

        if current_role != expected_role:
            return 0.0, [f"Turn {i+2}: Role mismatch. Expected '{expected_role}', but got '{current_role}'."]

        if current_role == "assistant":
            is_valid, reason = _validate_assistant_content(content)
            if not is_valid:
                return 0.0, [f"Turn {i+2} (assistant): Content format error - {reason}"]

            if "<result>" in content:
                expected_role = "end"
            else:
                expected_role = "user"

        elif current_role == "user":
            if "<tool_response>" not in content or "</tool_response>" not in content:
                return 0.0, [f"Turn {i+2} (user): Content is missing '<tool_response>' tags."]

            expected_role = "assistant"

    if expected_role != "end":
         return 0.0, ["Final State Error: Trajectory did not end with an assistant's <result>."]

    return 1.0, ["Perfect format!"]


if __name__ == '__main__':
    # Example 1: Perfect format
    valid_messages = [
        {"role": "system", "content": "..."}, {"role": "user", "content": "..."},
        {"role": "assistant", "content": "<thinking>I need to search.</thinking><tool_call>search(query='A')</tool_call>"},
        {"role": "user", "content": "<tool_response>Search result for A.</tool_response>"},
        {"role": "assistant", "content": "<thinking>Okay, I have the info.</thinking><result>The answer is B.</result>"},
    ]
    score, reasons = compute_format_score(valid_messages)
    print(f"--- Valid Example ---")
    print(f"Score: {score}")
    print(f"Reasons: {reasons}\n")

    # Example 2: Valid format with newlines
    valid_messages_with_newlines = [
        {"role": "system", "content": "..."}, {"role": "user", "content": "..."},
        {"role": "assistant", "content": "<thinking>I need to search.</thinking>\n\n  <tool_call>search(query='A')</tool_call>"},
        {"role": "user", "content": "\n<tool_response>Result</tool_response>\n"},
        {"role": "assistant", "content": "<thinking>Okay.</thinking>\n<answer>Done.</answer>"},
    ]
    score, reasons = compute_format_score(valid_messages_with_newlines)
    print(f"--- Valid with Newlines Example ---")
    print(f"Score: {score}")
    print(f"Reasons: {reasons}\n")

    # Example 3: Invalid role order
    invalid_role_messages = [
        {"role": "system", "content": "..."}, {"role": "user", "content": "..."},
        {"role": "assistant", "content": "<thinking>...</thinking><tool_call>...</tool_call>"},
        {"role": "assistant", "content": "<thinking>Oops</thinking><tool_call>...</tool_call>"},
    ]
    score, reasons = compute_format_score(invalid_role_messages)
    print(f"--- Invalid Role Example ---")
    print(f"Score: {score}")
    print(f"Reasons: {reasons}\n")
