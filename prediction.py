from src.model_io import load_model
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from src.config_loader import load_config

def main():
    # Loading the configuration
    config = load_config()

    # Loading the model
    try:
        model = load_model(config['save_model_path'])
    except FileNotFoundError:
        print (f"ERROR: File {config['save_model_path']} not found")
        print("Please run main.py first to train and save the model")
        return
    

    # Creating a DataFrame with a random generated value for each of the 30 features
    fake_data = np.random.rand(1,30)
    feature_names = load_breast_cancer().feature_names
    fake_df = pd.DataFrame(fake_data, columns=feature_names)

    # Predicting on this new data
    prediction = model.predict(fake_df)

    diagnosis = "Malignant" if prediction[0] == 0 else "Benign"

    print("\n"+"="*45)
    print(f"PREDICTION RESULT: {diagnosis.upper()}")
    print("="*45)


if  __name__ == "__main__":
    main()