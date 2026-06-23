from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from typing import Dict

def evaluate_model(X_test: pd.DataFrame, model: any, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluates the trained model accuracy

    Args:
    X_test, features of the test data
    model: Trained model (Random Forest Classifier)
    y_test, target labels of the test data

    Returns a dictionary contaning the evaluation metrics
    """
    # Model predicitons on test data
    predictions = model.predict(X_test)

    # Accuracy based on target results and model predictions
    accuracy = accuracy_score(y_test, predictions)

    print(f"The model as an accuracy of {accuracy*100:.2f}%")

    # More detailed report using scikitlearn function classification_report
    print("Classification report:")
    report = classification_report(y_test, predictions, target_names=["Malignant (0)", "Benign (1)"])
    print(report)

    return {"accuracy": accuracy }