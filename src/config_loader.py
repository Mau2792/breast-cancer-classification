import yaml
from typing import Dict, Any

def load_config(config_path: str="config.yaml") -> Dict[str, Any]:
    """
    Reads the YAML configuration file and return it as a dictionary
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        print("Configuration loaded successfully")
        return config
    
    except FileNotFoundError:
        print(f"ERROR: configuration file {config_path} not found")
        raise

    except yaml.YAMLError as error:
        print(f"ERROR: error parsing {config_path}: {error}")
        raise
