import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import joblib
import time
import os

# --- CONFIGURATION ---
DATA_FILE = 'grid_data.npy'


def train_svm():
    print("--- TRAINING SVM (LOCALLY) ---")
    if not os.path.exists(DATA_FILE):
        print("Error: grid_data.npy not found!")
        return

    data = np.load(DATA_FILE)

    # Use small slice for speed
    data_small = data[:10000]

    X = data_small[:, 1:]
    y = data_small[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[1/1] Training SVM...")
    start_time = time.time()

    svm_model = SVR(kernel='rbf')
    svm_model.fit(X_train, y_train)

    print(f"      SVM Training Time: {time.time() - start_time:.2f} seconds")

    joblib.dump(svm_model, 'svm_model.pkl')
    print("SUCCESS! 'svm_model.pkl' created.")


if __name__ == "__main__":
    train_svm()