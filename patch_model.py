import h5py

file_path = 'proposed_model.h5'

try:
    f = h5py.File(file_path, 'r+')
    model_config = f.attrs.get('model_config')

    if model_config is None:
        print("CRITICAL: No model_config found in the .h5 file.")
    else:
        # Check if it's bytes or already a string
        if isinstance(model_config, bytes):
            config_str = model_config.decode('utf-8')
        else:
            config_str = str(model_config)

        # Execute the surgical patch
        new_config_str = config_str.replace('"batch_shape":', '"batch_input_shape":')

        # Save it back in its original format
        if isinstance(model_config, bytes):
            f.attrs['model_config'] = new_config_str.encode('utf-8')
        else:
            f.attrs['model_config'] = new_config_str

        print("SUCCESS: Model architecture patched for TF 2.15 compatibility.")

    f.close()
except Exception as e:
    print(f"FAILED: {e}")