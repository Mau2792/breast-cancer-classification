from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from typing import Any

def train_model(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int, max_depth: int, random_state: int) -> Any:
    """
    Initializes and trains a Random Forest Classifier

    Args:
    - X_train, features of the training data
    - y_train, target labels of the training data
    - n_estimators, numbers of trees composing the forest
    - max_depth, maximum depth of each tree
    - random_state, control the randomness of the bootstrapping and features selection

    Returns the trained Random Forest Classifier
    """

    print(f"Initializing Random Forest with {n_estimators} having max depth of {max_depth}...")

    # Initializes the model (Random Forest Classifier)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    print("Training model...")

    # Trains the model with the given dataset
    model.fit(X_train, y_train)

    print("Training completed!")

    return model