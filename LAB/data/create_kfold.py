import os
import random
import math

# ==========================================
# Settings
# ==========================================
INPUT_FILE = "dataset_5000.txt"  # Ensure this matches your file name
OUTPUT_DIR = "kfold_data"
N_SPLITS = 5
SEED = 42  # Fixed seed for reproducibility

def create_kfold_datasets():
    # 1. Read the dataset
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please place it in the same directory.")
        return

    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        # Ignore empty lines
        lines = [line.strip() for line in f if line.strip()]

    total_samples = len(lines)
    print(f"Total samples: {total_samples}")

    if total_samples == 0:
        print("Error: The dataset is empty.")
        return

    # 2. Shuffle data (Randomize)
    random.seed(SEED)
    random.shuffle(lines)

    # 3. Create Output Directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # 4. Create Folds
    fold_size = math.ceil(total_samples / N_SPLITS)

    for i in range(N_SPLITS):
        # Calculate indices for test data
        start_idx = i * fold_size
        end_idx = min((i + 1) * fold_size, total_samples)

        # Split data
        test_data = lines[start_idx:end_idx]
        train_data = lines[:start_idx] + lines[end_idx:]

        # Create directory for this fold
        fold_dir_name = f"fold_{i+1}"
        fold_path = os.path.join(OUTPUT_DIR, fold_dir_name)
        
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)

        # Write files
        train_file_path = os.path.join(fold_path, "train.txt")
        test_file_path = os.path.join(fold_path, "test.txt")

        with open(train_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_data) + '\n')
        
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_data) + '\n')

        print(f"[{fold_dir_name}] Created.")
        print(f"  - Train: {len(train_data)} samples")
        print(f"  - Test : {len(test_data)} samples")

    print("\nSuccessfully created 5-fold datasets in 'kfold_data/' directory.")

if __name__ == "__main__":
    create_kfold_datasets()