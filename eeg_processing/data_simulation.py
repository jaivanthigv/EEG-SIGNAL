import numpy as np

def simulate_eeg_data():
    """
    Simulate EEG data with 64 channels and 500 time samples.
    """
    data = np.random.rand(64, 500)  # Simulate 64 channels, 500 samples
    return data