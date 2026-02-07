import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# CONFIGURATION
DATA_FILE = 'household_power_consumption.txt'
OUTPUT_FILE = 'grid_data.npy'


def process_data():
    print(f"--- STARTING DATA PIPELINE ---")

    # 1. Check if file exists
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] Could not find '{DATA_FILE}'. Please download it from UCI Repository.")
        return

    print(f"[1/4] Loading dataset (this may take a moment)...")
    # The dataset uses ';' as a separator and has '?' for missing values
    df = pd.read_csv(DATA_FILE, sep=';',
                     parse_dates={'dt': ['Date', 'Time']},
                     infer_datetime_format=True,
                     low_memory=False,
                     na_values=['nan', '?'])

    print(f"[2/4] Cleaning data...")
    # Drop rows with missing values (NaN)
    original_len = len(df)
    df.dropna(inplace=True)
    new_len = len(df)
    print(f"      Removed {original_len - new_len} rows with missing values.")

    # 2. Feature Selection
    # Based on the paper, we need Power (P), Intensity, and Voltage
    # We will pick the first 7 columns for training
    # columns: Global_active_power, Global_reactive_power, Voltage, Global_intensity, Sub_metering_1, 2, 3
    cols_to_keep = ['Global_active_power', 'Global_reactive_power',
                    'Voltage', 'Global_intensity',
                    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

    dataset = df[cols_to_keep].values.astype('float32')

    print(f"[3/4] Normalizing data (scaling between 0 and 1)...")
    # This is crucial for Neural Networks (LSTM/CNN) to converge
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)

    print(f"[4/4] Saving processed file to '{OUTPUT_FILE}'...")
    # We save as .npy because it's faster and smaller than CSV
    np.save(OUTPUT_FILE, dataset_scaled)

    print(f"--- SUCCESS! Data ready for AI Models. ---")
    print(f"File saved: {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    process_data()