import os
import json
from typing import List, Dict, Any, Tuple

def load_data(data_folder: str) -> List[Dict[str, Any]]:
    """Load JSON files from the specified data folder."""
    all_data = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    all_data.append(data)
    return all_data


def extract_pairs(json_data: Dict[str, Any]) -> List[Tuple[List[List[int]], List[List[int]]]]:
    """Extract training input-output pairs from the JSON data."""
    train_pairs = [
        (entry["input"], entry["output"]) for entry in json_data.get("train", [])
    ]
    return train_pairs
