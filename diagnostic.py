import numpy as np
from tensorflow.keras.models import load_model

print("--- LOADING DATA ---")
try:
    data = np.load('grid_data.npy')
    print(f"Data Shape: {data.shape}")
    print(f"First 5 values: \n{data[:5]}")
except Exception as e:
    print(f"Failed to load data: {e}")

print("\n--- LOADING MODEL ---")
try:
    model = load_model('proposed_model.h5')
    print("Model Input Shape Expected:")
    print(model.input_shape)
except Exception as e:
    print(f"Failed to load model: {e}")