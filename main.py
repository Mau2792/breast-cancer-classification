from src.config_loader import load_config

def main():
    print("Starting Breast Cancer Pipeline")
    
    # Load the configuration
    config = load_config()

    # Print a value to see if it works
    print(f"Using {config['n_estimators']} trees for the Random Forest")
    print(f"Raw data are saved in {config['raw_data_path']}")

if __name__ == "__main__":
    main()