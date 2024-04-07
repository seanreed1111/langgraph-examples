import json
import os
from pathlib import Path

LOCAL_CONFIG_DIR = Path.home() / "WAB-config"

def run_azure_config(config_dir):
    all_config_file_path = config_dir / "allconfig.json"
    config = {}
    with open(all_config_file_path) as json_config:
        config.update(json.load(json_config))
        for k in config:
            os.environ[k] = config[k]
    return True