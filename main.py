from src.config_loader import load_config
from src.data_preprocessing import fetch_and_save_raw_data, split_data

def main():
    print("Starting Breast Cancer Pipeline")
    
    # Load the configuration
    config = load_config()

    print(f"DEBUG: Il percorso è: {config['raw_data_path']}")

    # Print a value to see if it works
    print(f"Using {config['n_estimators']} trees for the Random Forest")
    print(f"Raw data are saved in {config['raw_data_path']}")

    # Saving raw data in pandas DataFrame
    raw_df = fetch_and_save_raw_data(config['raw_data_path'])

    # Splits train and test sets
    X_train, X_test, y_train, t_test = split_data(
        df=raw_df,
        test_size=config['test_size'],
        random_state=config['random_state']
    )

    print("----Data Preprocessing successfully completed----")


if __name__ == "__main__":
    main()