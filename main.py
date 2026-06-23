from src.config_loader import load_config
from src.data_preprocessing import fetch_and_save_raw_data, split_data
from src.model_training import train_model
from src.evaluation import evaluate_model

def main():
    print("Starting Breast Cancer Pipeline")
    
    # Load the configuration
    config = load_config()


    # Saving raw data in pandas DataFrame
    raw_df = fetch_and_save_raw_data(config['raw_data_path'])

    # Splitting train and test sets
    X_train, X_test, y_train, y_test = split_data(
        df=raw_df,
        test_size=config['test_size'],
        random_state=config['random_state']
    )

    print("----Data Preprocessing successfully completed----")

    # Initializing and traininig model (Random Forest Classifier)
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        n_estimators=config['n_estimators'],
        max_depth=config['depth'],
        random_state=config['random_state']
    )
     
    # Testing the model and evaluating accuracy
    report = evaluate_model(
        X_test=X_test,
        model=model,
        y_test=y_test
    )


    

if __name__ == "__main__":
    main()