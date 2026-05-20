from sklearn.decomposition import PCA

def extract_features(eeg_data):
    """
    Extract features from EEG data using PCA.
    """
    pca = PCA(n_components=5)  # Reduce to 5 components for simplicity
    eeg_data_reduced = pca.fit_transform(eeg_data.T).T
    return eeg_data_reduced
