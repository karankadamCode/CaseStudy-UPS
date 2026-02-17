"""
prompt_manager.py

Purpose:
This module is responsible for loading and validating prompt definitions
stored in YAML files. It ensures that required prompt fields (system and user)
are present and correctly typed before being used by the RAG pipeline.

Usage:
- Used by the RAG query engine to load system prompts and user templates
- Supports prompt versioning via YAML files
- Provides strict validation to avoid silent prompt failures at runtime

Author:
Karan Kadam
"""

import os
import yaml
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("prompt_manager")


class PromptLoadError(RuntimeError):
    """
    Raised when a prompt YAML file is invalid, malformed,
    or missing required fields.
    """
    pass


def load_prompt_yaml(path: str) -> Dict[str, Any]:
    """
    Load and validate a prompt YAML file.

    Validates:
    - File exists
    - YAML is a mapping/object
    - Contains a 'prompt' object
    - 'prompt.system' and 'prompt.user' are non-empty strings

    Args:
        path: Absolute or relative path to the YAML prompt file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        ValueError: If path is invalid.
        FileNotFoundError: If file does not exist.
        PromptLoadError: If YAML structure or required fields are invalid.
    """
    
    if not path or not isinstance(path, str):
        raise ValueError("Prompt YAML path must be a non-empty string")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt YAML not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.exception("Failed to read/parse prompt YAML")
        raise PromptLoadError("Prompt YAML load failed") from e

    if not isinstance(data, dict):
        raise PromptLoadError("Prompt YAML must be a YAML mapping/object")

    prompt_obj = data.get("prompt")
    if not isinstance(prompt_obj, dict):
        raise PromptLoadError("Prompt YAML missing 'prompt' object")

    system_prompt = prompt_obj.get("system")
    user_template = prompt_obj.get("user")

    if not system_prompt or not isinstance(system_prompt, str):
        raise PromptLoadError("Prompt YAML missing required field: prompt.system (string)")

    if not user_template or not isinstance(user_template, str):
        raise PromptLoadError("Prompt YAML missing required field: prompt.user (string)")

    return data


def get_prompts(path: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Load prompt YAML and return the system prompt, user template,
    and full prompt specification.

    Args:
        path: Path to the prompt YAML file.

    Returns:
        A tuple of:
        - system_prompt (str)
        - user_template (str)
        - full prompt specification (dict)
    """
    logger.info(f"Loading prompt spec from: {path}")
    spec = load_prompt_yaml(path)

    system_prompt = spec["prompt"]["system"]
    user_template = spec["prompt"]["user"]

    return system_prompt, user_template, spec