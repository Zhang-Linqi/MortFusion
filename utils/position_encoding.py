# -*- coding: utf-8 -*-


import numpy as np
import torch

def get_sincos_positional_encoding(num_positions, d_model):
    position = np.arange(num_positions)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((num_positions, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe 
