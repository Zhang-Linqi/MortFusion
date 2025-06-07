# -*- coding: utf-8 -*-
# predictor.py

import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from configs import config 
from utils.position_encoding import get_sincos_positional_encoding  

def recursive_predict_and_export(
    model,
    countries,
    H_tensor,
    mortality_data,
    OUTPUT_DIR,
    age_range,
    year_range,
    future_years,
    TRANSFORMER_INPUT_DIM,
    model_type="encoderonly",
    target_countries=None,
    loader=None
):

    num_countries = len(countries)
    num_ages = len(age_range)
    seq_len_hist = len(year_range)
    seq_len_total = seq_len_hist + len(future_years)

    # 全体预测存储
    all_preds_extended = np.zeros((num_countries, num_ages, seq_len_total))
    all_preds_extended[:, :, :seq_len_hist] = mortality_data

    if target_countries is not None:
        if isinstance(target_countries, str):
            target_countries = [target_countries]
        target_idxs = [countries.index(c) for c in target_countries]
    else:
        target_idxs = range(num_countries)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for idx in tqdm(target_idxs, desc="递推预测并导出"):
        country_code = countries[idx]
        struct_feat = H_tensor[idx].cpu().numpy()  # [64]
        input_country = np.zeros((num_ages, seq_len_total, TRANSFORMER_INPUT_DIM))
        input_country[:, :seq_len_hist, 0] = mortality_data[idx]
        
        # === 结构特征和位置编码拼接 ===
        struct_feat_expanded = np.broadcast_to(struct_feat.reshape(1, 1, -1), (num_ages, seq_len_total, struct_feat.shape[0]))
        pe = get_sincos_positional_encoding(seq_len_total, config.POS_DIM)  # [seq_len_total, pos_dim]
        pe = np.broadcast_to(pe, (num_ages, seq_len_total, config.POS_DIM))
        struct_and_pos = np.concatenate([struct_feat_expanded, pe], axis=-1)  # [num_ages, seq_len_total, 64+POS_DIM]
        input_country[:, :, 1:] = struct_and_pos

        if model_type == "encoderonly":
            # === Encoder-only递推 ===
            for i, year in enumerate(future_years):
                this_year_idx = seq_len_hist + i
                flat_input = input_country[:, :this_year_idx, :].reshape(1, -1, input_country.shape[2])
                x_pred = torch.tensor(flat_input, dtype=torch.float32).to(next(model.parameters()).device)
                pred_seq = model(x_pred)
                pred_year = pred_seq[0, -num_ages:, 0].detach().cpu().numpy()
                input_country[:, this_year_idx, 0] = pred_year
                all_preds_extended[idx, :, this_year_idx] = pred_year
        elif model_type == "seq2seq":
            # === Seq2Seq并行递推/并行预测未来 ===
            src_input = input_country[:, :seq_len_hist, :]
            # decoder输入：用历史最后一年和全0拼到未来区间
            first_dec = input_country[:, seq_len_hist-1:seq_len_hist, :]
            future_zeros = np.zeros((num_ages, len(future_years)-1, TRANSFORMER_INPUT_DIM))
            decoder_input = np.concatenate([first_dec, future_zeros], axis=1)
            src_input = src_input.reshape(1, *src_input.shape)
            decoder_input = decoder_input.reshape(1, *decoder_input.shape)
            x_src = torch.tensor(src_input, dtype=torch.float32).to(next(model.parameters()).device)
            x_dec = torch.tensor(decoder_input, dtype=torch.float32).to(next(model.parameters()).device)
            pred_seq = model(x_src, x_dec)
            pred_all = pred_seq[0, :, 0].detach().cpu().numpy()
            input_country[:, seq_len_hist:, 0] = pred_all.T
            all_preds_extended[idx, :, seq_len_hist:] = pred_all.T
        else:
            raise ValueError(f"未知模型类型: {model_type}")
        # 保存单国csv
        df_pred = pd.DataFrame(all_preds_extended[idx], index=age_range, columns=year_range+future_years)
       # === 反标准化与反 log 处理 ===
        if loader is not None:
            mean, std = loader.stats[config.GENDER][country_code]
            mean = mean.reindex(df_pred.index)
            std = std.reindex(df_pred.index)
            df_pred_restored = df_pred.mul(std, axis=0).add(mean, axis=0)
            df_pred_mortality = np.exp(df_pred_restored)
            df_pred_mortality.to_csv(os.path.join(OUTPUT_DIR, f"{country_code}_pred_mortality.csv"))
        df_pred.to_csv(os.path.join(OUTPUT_DIR, f"{country_code}_pred_extended.csv"))
        print(f"{country_code} 保存完成！")
    return all_preds_extended
