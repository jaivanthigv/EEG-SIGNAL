from scipy.signal import butter, lfilter

def bandpass_filter(data, lowcut=1, highcut=50, fs=128, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)
