import numpy as np

def extract_features(eeg_signal):
    # Simple example: mean, std, max, min, etc.
    mean = np.mean(eeg_signal)
    std = np.std(eeg_signal)
    max_val = np.max(eeg_signal)
    min_val = np.min(eeg_signal)

    # Pad to 10 features if needed
    features = [mean, std, max_val, min_val]
    while len(features) < 10:
        features.append(0)  # padding

    return np.array(features).reshape(1, -1)