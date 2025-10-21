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

"""Centralized environment variable configuration for tool server."""

import os
from typing import Any, Dict


class EnvironmentConfig:
    """Centralized environment variable management for tool server."""

    def __init__(self):
        """Initialize configuration by loading environment variables."""
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load all environment variables with defaults.

        Returns:
            Dictionary containing all configuration values.
        """
        return {
            # Server configuration
            "MAX_CONCURRENT_REQUESTS": int(os.getenv("MAX_CONCURRENT_REQUESTS", "2000")),
            "REQUEST_TIMEOUT": int(os.getenv("REQUEST_TIMEOUT", "180")),
            "VISIT_SEMAPHORE_LIMIT": int(os.getenv("VISIT_SEMAPHORE_LIMIT", "200")),
            "SEARCH_SEMAPHORE_LIMIT": int(os.getenv("SEARCH_SEMAPHORE_LIMIT", "500")),

            # API Keys
            "WEBCONTENT_MAXLENGTH": int(os.getenv("WEBCONTENT_MAXLENGTH", "150000")),
            "JINA_API_KEY": os.getenv("JINA_API_KEY"),
            "GOOGLE_SEARCH_KEY": os.getenv("GOOGLE_SEARCH_KEY"),
            "TOOL_SERVER_LLM_API_KEY": os.getenv("TOOL_SERVER_LLM_API_KEY"),
            "TOOL_SERVER_LLM_BASE_URL": os.getenv("TOOL_SERVER_LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            "TOOL_SERVER_LLM_MODEL": os.getenv("TOOL_SERVER_LLM_MODEL", "deepseek/deepseek-chat-v3-0324"),

            # Search configuration
            "SEARCH_MAX_QUERIES": int(os.getenv("SEARCH_MAX_QUERIES", "3")),
            "IGNORE_JINA": os.getenv("IGNORE_JINA", "false").lower() == "true",

            # Server settings
            "HOST": os.getenv("TOOL_SERVER_HOST", "0.0.0.0"),
            "PORT": int(os.getenv("TOOL_SERVER_PORT", "50001")),
            "WORKERS": 16,
            "BACKLOG": 2048,
            "LIMIT_CONCURRENCY": 2000,
            "LIMIT_MAX_REQUESTS": 10000,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key to retrieve.
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        return self._config.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values.

        Returns:
            Copy of all configuration values.
        """
        return self._config.copy()

    def get_api_keys_status(self) -> Dict[str, bool]:
        """Get status of API key configuration.

        Returns:
            Dictionary mapping API key names to their presence status.
        """
        return {
            "google_search": bool(self._config.get("GOOGLE_SEARCH_KEY")),
            "jina": bool(self._config.get("JINA_API_KEY")),
            "llm_provider": bool(self._config.get("TOOL_SERVER_LLM_API_KEY"))
        }

    def validate_required_keys(self) -> Dict[str, bool]:
        """Validate that required API keys are present.

        Returns:
            Dictionary mapping API key names to validation status.
        """
        required_keys = {
            "google_search": self._config.get("GOOGLE_SEARCH_KEY"),
            "jina": self._config.get("JINA_API_KEY"),
            "llm_provider": self._config.get("TOOL_SERVER_LLM_API_KEY")
        }

        return {
            key: bool(value) for key, value in required_keys.items()
        }


# Global configuration instance
config = EnvironmentConfig()
