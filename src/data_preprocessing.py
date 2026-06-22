import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from typing import Tuple

def fetch_and_save_raw_data(raw_data_path: str) -> pd.DataFrame:
    """
    Fetches raw data from official SciKitLearn dataset, converts it to
    a pandas DataFrame and saves it as a csv file at the specified path
    """
    # Loads data and converts to pandas data frame
    dataset = load_breast_cancer()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

    # Adds results of  the diagnosis to our df
    df['target'] = dataset.target

    # Creates a directory to save csv file, if it doesn't exist already
    directory = os.path.dirname(raw_data_path)
    os.makedirs(directory, exist_ok=True)

    # Converts the DataFrame to csv and  saves it in the specified directory
    # index=False prevents pandas from writing the row numbers in the file
    df.to_csv(raw_data_path, index=False)

    print(f"Raw data succesfully saved to: {raw_data_path}")
    return df


def split_data(df: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data frame in test data and train data according to test_size.
    Returns X_train, X_test, y_train, y_test
    """
    print("Splitting data in train and test sets...")
    
    # Splits data and results
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Performs a further split using sklearn function train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {y_test.shape[0]}")

    return [X_train, X_test, y_train, y_test]

    




