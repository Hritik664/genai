import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Step 1: Create folder structure if not already created
def create_folders():
    base_dir = os.getenv("RAW_DATA_PATH", "../data")
    
    # Create directories for raw and processed data if they don't exist
    os.makedirs(os.path.join(base_dir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'processed'), exist_ok=True)
    print(f"Folder structure created at: {base_dir}")
    
# Step 2: Load and combine datasets (NSL-KDD and UNSW-NB15)
def load_and_combine_data():
    raw_data_path = os.getenv("RAW_DATA_PATH", "data/raw")
    print(f"Raw data path: {raw_data_path}")

    
    # Load the raw datasets
    nsl_kdd_train = pd.read_csv(os.path.join(raw_data_path, "kdd_train.csv"))
    nsl_kdd_test = pd.read_csv(os.path.join(raw_data_path, "kdd_test.csv"))
    unsw_nb15_train = pd.read_csv(os.path.join(raw_data_path, "UNSW_NB15_training-set.csv"))
    unsw_nb15_test = pd.read_csv(os.path.join(raw_data_path, "UNSW_NB15_testing-set.csv"))
    
    # Combine datasets
    combined_train = pd.concat([nsl_kdd_train, unsw_nb15_train], ignore_index=True)
    combined_test = pd.concat([nsl_kdd_test, unsw_nb15_test], ignore_index=True)
    
    print("Data combined successfully.")
    return combined_train, combined_test

# Step 3: Separate features and labels, and balance the training data
def balance_data(combined_train, combined_test):
    # Separate features and labels for balancing
    X_train = combined_train.drop('label', axis=1)  # Replace 'label' with actual label column name
    y_train = combined_train['label']
    
    # Balance the dataset by resampling (upsampling the minority class)
    majority_class = combined_train[combined_train['label'] == 0]  # Adjust class label values as per dataset
    minority_class = combined_train[combined_train['label'] == 1]

    minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
    balanced_train = pd.concat([majority_class, minority_upsampled])

    # Separate balanced features and labels
    X_train_balanced = balanced_train.drop('label', axis=1)
    y_train_balanced = balanced_train['label']

    print("Training data balanced successfully.")
    return X_train_balanced, y_train_balanced, combined_test

# Step 4: Save the processed data to the processed folder
def save_processed_data(X_train_balanced, y_train_balanced, combined_test):
    processed_data_path = os.getenv("PROCESSED_DATA_PATH", "../data/processed")
    
    # Save balanced data and test data
    X_train_balanced.to_csv(os.path.join(processed_data_path, "X_train_combined.csv"), index=False)
    y_train_balanced.to_csv(os.path.join(processed_data_path, "y_train_combined.csv"), index=False)
    combined_test.to_csv(os.path.join(processed_data_path, "X_test_combined.csv"), index=False)
    
    print("Processed data saved successfully.")

# Step 5: Split Data for Cross-Validation (Optional)
def split_data_for_validation(X_train_balanced, y_train_balanced):
    processed_data_path = os.getenv("PROCESSED_DATA_PATH", "../data/processed")
    
    # Split the balanced training data into training and validation sets (80% training, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train_balanced, y_train_balanced, test_size=0.2, random_state=42)
    
    # Save the validation set
    X_train.to_csv(os.path.join(processed_data_path, "X_train_final.csv"), index=False)
    y_train.to_csv(os.path.join(processed_data_path, "y_train_final.csv"), index=False)
    X_val.to_csv(os.path.join(processed_data_path, "X_val.csv"), index=False)
    y_val.to_csv(os.path.join(processed_data_path, "y_val.csv"), index=False)
    
    print("Data split for validation saved successfully.")
    return X_train, y_train, X_val, y_val

# Step 6: Main function to execute the preprocessing steps
def main():
    # Step 1: Create necessary folders
    create_folders()
    
    # Step 2: Load and combine the datasets
    combined_train, combined_test = load_and_combine_data()
    
    # Step 3: Balance the training data
    X_train_balanced, y_train_balanced, combined_test = balance_data(combined_train, combined_test)
    
    # Step 4: Save the processed datasets
    save_processed_data(X_train_balanced, y_train_balanced, combined_test)
    
    # Step 5: Split the data for cross-validation (Optional)
    X_train, y_train, X_val, y_val = split_data_for_validation(X_train_balanced, y_train_balanced)

# Run the preprocessing script
if __name__ == "__main__":
    main()
