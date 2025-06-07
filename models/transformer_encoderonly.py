# -*- coding: utf-8 -*-


# models/transformer.py

import torch
import torch.nn as nn
from configs import config
class MortalityTransformer(nn.Module):
    def __init__(self, input_dim=65, d_model=128, nhead=8, num_layers=4, dropout=0.1, output_dim=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 多层线性层替代Decoder
        dims = config.DECODER_HIDDEN_DIMS  
        if config.DECODER_ACTIVATION.lower() == 'relu':
            activation = nn.ReLU()
        elif config.DECODER_ACTIVATION.lower() == 'gelu':
            activation = nn.GELU()
        else:
            raise ValueError("激活函数只支持 relu 或 gelu")

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # 最后一层不要激活
                layers.append(activation)
        self.output_head = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]
        return: [batch, seq_len, output_dim]
        """
        x = self.input_proj(x)
        x = self.transformer(x)
        out = self.output_head(x)
        return out

def masked_mse_loss(pred, target, mask):
    mse = (pred - target) ** 2
    masked = mse.squeeze(-1) * mask
    return masked.sum() / mask.sum()
