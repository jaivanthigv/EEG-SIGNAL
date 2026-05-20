from sklearn.preprocessing import StandardScaler

def preprocess_eeg_data(eeg_data):
    """
    Preprocess EEG data by standardizing it.
    """
    scaler = StandardScaler()
    eeg_data_scaled = scaler.fit_transform(eeg_data.T).T
    return eeg_data_scaled
