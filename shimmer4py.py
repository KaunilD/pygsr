import numpy as np
import argparse
import csv

def read_data(file_path: str) -> np.ndarray:
    """
        Returns a squeezed row vector.
    """
    a = []
    with open(file_path, 'r') as cvs_file:
        csv_reader = csv.reader(cvs_file)
        next(csv_reader)
        for row in csv_reader:
            a.append(float(row[1]))
    
    return np.asarray(a, dtype=np.float32).reshape((-1))

def get_mean(data_slice: np.ndarray) -> float:
    """
       Returns average of the data slice along the first (and only)
       dimension. 
    """
    res = [i for i in data_slice]
    sample_size = data_slice.shape[0]

    return np.sum(res, axis=0)/sample_size

def get_max(data_slice: np.ndarray) -> (int, float):
    """
       Returns a tuple containing index where the maximum
       amplitude occurs in and it's value
    
    """
    idx = np.argmax(data_slice, axis=0)
    return (idx, data_slice[idx])

def get_min(data_slice: np.ndarray) -> (int, float):
    """
        Returns a tuple containing index where the least 
        amplitude falls in and it's value
    """
    idx = np.argmin(data_slice, axis=0)
    return (idx, data_slice[idx])

def get_sd(data_slice: np.ndarray, sample_size: int) -> float:
    """
        Kurtosis uses N for calculating for calculating
        standar deviation instead of N-1.
    """
    mean    = get_mean(data_slice)
    res     = np.asarray([ ((i - mean)**2) for i in data_slice])

    return np.sqrt(get_mean(res))
    
def get_kurtosis(data_slice: np.ndarray) -> float:
    std         = get_sd(data_slice, data_slice.shape[0])
    mean        = get_mean(data_slice)
    res         = np.asarray([(i - mean)**4 for i in data_slice])
    return get_mean(res)/(std**4)

def get_alsc(data_slice: np.ndarray) -> float:
    res = [
        np.sqrt(1+(data_slice[idx] - data_slice[idx-1])**2) 
        for idx, i in enumerate(data_slice[1:])
    ]
    return np.sum(res)

def get_features(data_slice: np.ndarray) -> dict:
    return {
        "max":  get_max(data_slice),
        "min":  get_min(data_slice),
        "mean": get_mean(data_slice),
        "sd":   get_sd(data_slice, data_slice.shape[0]-1),
        "kurt": get_kurtosis(data_slice),
        "alsc": get_alsc(data_slice)
    }
