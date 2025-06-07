# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
from glob import glob
from typing import Tuple, List, Optional, Dict
from utils.data_loader import MortalityDataLoader
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Part 1: GCN 模型定义
# -----------------------------
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        # x: [N, F], adj: [N, N]
        support = self.linear(x)         # [N, F_out]
        output = torch.matmul(adj, support)  # [N, F_out]
        return output

class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(GCNEncoder, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # 第一层
        x = self.dropout(x)
        x = self.gc2(x, adj)          # 第二层
        return x

# -----------------------------
# Part 2: 构建国家级输入特征矩阵
# -----------------------------
def build_input_matrix_from_ready_features(
    folder_path: str,
    gender: str = 'female',
    n_components: int = 3
) -> Tuple[np.ndarray, List[str]]:
    """
    读取主成分后的国家级特征，生成输入特征矩阵 X。
    每个文件 shape=(主成分数, 年龄数)，默认使用前 n_components 个主成分。
    """
    X = []
    country_list = []
    gender_folder = os.path.join(folder_path, gender)
    files = sorted(glob(os.path.join(gender_folder, "*.csv")))

    for file_path in files:
        try:
            df = pd.read_csv(file_path, index_col=0)
            if df.shape[0] < n_components:
                raise ValueError(f"{file_path} 行数 < {n_components}，主成分不足")
            vec = df.values[:n_components, :].flatten()  # shape: [n_components * 年龄数]
            X.append(vec)

            country_code = os.path.basename(file_path).split("_")[0]
            country_list.append(country_code)

        except Exception as e:
            print(f" 跳过 {file_path}，错误: {e}")

    X = np.stack(X)  # shape: [国家数, n_components * 年龄数]
    return X, country_list

def masked_mse_loss(pred, target, mask):
    mse = (pred - target) ** 2
    masked = mse.squeeze(-1) * mask
    return masked.sum() / mask.sum()


