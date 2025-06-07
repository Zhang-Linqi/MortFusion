# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class MortalityTransformerSeq2Seq(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dropout=0.1, output_dim=1, max_seq_len=2048):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_head = nn.Linear(d_model, output_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, src, tgt):
        # src, tgt: [batch, seq_len, input_dim]
        x_src = self.input_proj(src) + self.pos_embed[:, :src.size(1), :]
        memory = self.encoder(x_src)
        x_tgt = self.input_proj(tgt) + self.pos_embed[:, :tgt.size(1), :]
        out = self.decoder(x_tgt, memory)
        return self.output_head(out)