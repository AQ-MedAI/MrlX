"""
Module: config
--------------
Unified environment variable configuration management.
Loads environment variables and provides centralized configuration access for the project.
"""

import os


class Config:
    """
    Environment variable configuration class.

    Attributes:
        MAX_TURNS (int): Maximum number of conversation turns for doctor-patient simulation.
        MAX_MODEL_LEN (int): Maximum allowed token length for model inputs.
        PATIENT_MAX_NEW_TOKENS (int): Maximum new tokens to generate in patient model.
        PATIENT_IP (str): IP address of patient model endpoint.
        PATIENT_TOKENIZER_PATH (str): Path to patient model's tokenizer.
        KEY_SUFFIX (str): Suffix used for generating unique database keys.
        DEEPSEEK_R1_API_KEY (str): API key for DeepSeek-R1 model.
    """

    # Maximum conversation turns
    MAX_TURNS = int(os.getenv("MAX_TURNS", "15"))

    # Maximum model length
    MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))

    # Patient model configuration
    PATIENT_MAX_NEW_TOKENS = int(os.getenv("PATIENT_MAX_NEW_TOKENS", "1024"))
    PATIENT_IP = os.getenv("PATIENT_IP")
    PATIENT_TOKENIZER_PATH = os.getenv(
        "PATIENT_TOKENIZER_PATH",
        "",
    )

    # Database ip configuration
    DATABASE_SERVER_IP = os.getenv("DATABASE_SERVER_IP")

    # Database configuration
    KEY_SUFFIX = os.getenv("KEY_SUFFIX")

    # DeepSeek API configuration
    DEEPSEEK_R1_API_KEY = os.getenv("DEEPSEEK_R1_API_KEY", "EMPTY")
    DEEPSEEK_R1_BASE_URL = os.getenv(
        "DEEPSEEK_R1_BASE_URL", ""
    )

    # Default reward score for aborted or failed samples
    DEFAULT_SCORE = float(os.getenv("DEFAULT_SCORE", "0.0"))

    def __init__(self):
        """Print configuration content on initialization"""
        print("=" * 50)
        print("Global Configuration Loaded:")
        print(f"  MAX_TURNS: {self.MAX_TURNS}")
        print(f"  MAX_MODEL_LEN: {self.MAX_MODEL_LEN}")
        print(f"  PATIENT_MAX_NEW_TOKENS: {self.PATIENT_MAX_NEW_TOKENS}")
        print(f"  DATABASE_SERVER_IP: {self.DATABASE_SERVER_IP}")
        print(f"  PATIENT_IP: {self.PATIENT_IP}")
        print(f"  PATIENT_TOKENIZER_PATH: {self.PATIENT_TOKENIZER_PATH}")
        print(f"  KEY_SUFFIX: {self.KEY_SUFFIX}")
        print(
            f"  DEEPSEEK_R1_API_KEY: {'Set' if self.DEEPSEEK_R1_API_KEY != 'EMPTY' else 'Not Set'}"
        )
        print(f"  DEFAULT_SCORE: {self.DEFAULT_SCORE}")
        print("=" * 50)


# Create global configuration instance
global_config = Config()
