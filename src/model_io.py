import os
import joblib
from typing import Any

def save_model(model: Any, path: str) -> None:
    """
    Saves the trained model in the specified path
    """

    # Extracts directory name out of full path
    directory = os.path.dirname(path)

    # Creates the directory if it doesn't exist
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Serializes and saves the model
    joblib.dump(model, path)
    print(f"Model successfully saved to {path}")


def load_model(path: str) -> Any:
    """
    Extracts annd loads model stored ad the specified path
    """
    # If the path does not exist return error
    if not os.path.exists(path):
        raise FileNotFoundError
    
    print(f"Loading model {path}")
    
    # Deserializes and returns the model
    return joblib.load(path)