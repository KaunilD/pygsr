from shimmer4py import *
if __name__=="__main__":
    fs = 10 # heartz
    win_size = 5 # seconds
    stride = 1 # seconds

    data = read_data('data/GSR_calibrated_signal.csv')

    features = []

    idx = 0
    while idx < data.shape[0]:
        features.append(get_features(data[idx:idx+win_size*fs]))
        idx+= stride*fs

    print(features[0])