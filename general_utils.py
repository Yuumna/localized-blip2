import os
import json
from typing import Dict
from pathlib import Path


def load_jsonl(file):
    data_str = Path(file).read_text(encoding="utf-8")
    data = []
    for i, line in enumerate(data_str.splitlines()):
        try:
            data.append(json.loads(line))
        except Exception as e:
            raise RuntimeError(
                f"Error loading json file {file} line {i}") from e
    return data


def get_env_variables() -> Dict:
    """Get environment variables for training and evaluation.

    Returns:
        Dict: dictionary of environment variables
    """
    env_variables = {}

    try:
        env_variables["STORAGE_DIR"] = os.environ["STORAGE_DIR"]
    except KeyError:
        raise RuntimeError("Please set the environment variable by running init_gcloud_env.sh")

    return env_variables