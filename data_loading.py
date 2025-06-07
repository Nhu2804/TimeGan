"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import numpy as np


def MinMaxScaler(data):
    """Min Max normalizer.
    Args:
      - data: original data
    Returns:
      - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def sine_data_generation(no, seq_len, dim):
    """Sine data generation.
    Args:
      - no: number of samples
      - seq_len: sequence length
      - dim: feature dimensions
    Returns:
      - data: list of np.arrays (each [seq_len, dim])
    """  
    data = []
    for _ in range(no):      
        temp = []
        for k in range(dim):
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)
        temp = np.transpose(np.asarray(temp))
        temp = (temp + 1) * 0.5  # normalize to [0, 1]
        data.append(temp)
    return data


def real_data_loading(data_name, seq_len):
    """Load and preprocess real-world datasets.
    Args:
      - data_name: stock or energy
      - seq_len: sequence length
    Returns:
      - data: list of np.arrays (each [seq_len, dim])
    """  
    assert data_name in ['stock', 'energy']

    if data_name == 'stock':
        ori_data = np.loadtxt('data/stock_data.csv', delimiter=",", skiprows=1)
    elif data_name == 'energy':
        ori_data = np.loadtxt('data/energy_data.csv', delimiter=",", skiprows=1)

    ori_data = ori_data[::-1]
    ori_data = MinMaxScaler(ori_data)

    temp_data = []
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    idx = np.random.permutation(len(temp_data))
    data = [temp_data[i] for i in idx]

    return data