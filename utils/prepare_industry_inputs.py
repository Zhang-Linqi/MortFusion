# -*- coding: utf-8 -*-

import torch
import pandas as pd
import numpy as np
from models.mortality_gcn import build_input_matrix_from_ready_features, GCNEncoder
from utils.position_encoding import get_sincos_positional_encoding

def prepare_industry_inputs(config, industry_df):

    # === 获取中国结构特征 ===
    # 1. 用主模型生成 h_CN（GCNEncoder）
    X_np, countries = build_input_matrix_from_ready_features(
        folder_path=config.PCA_FOLDER,
        gender=config.GENDER,
        n_components=config.N_COMPONENTS
    )
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    cn_idx = countries.index("CN")

    df = pd.read_csv(config.DTW_MATRIX_PATH, header=0, index_col=0)
    dtw_country_names = list(df.index)
    if countries != dtw_country_names:
        df = df.loc[countries, countries]  
    A = df.values.astype(float)
    A_sim = np.exp(-A / A.std())  
    np.fill_diagonal(A_sim, 1)    
    A_tensor = torch.tensor(A_sim, dtype=torch.float32)

    gcn_model = GCNEncoder(
        input_dim=X_tensor.shape[1], 
        hidden_dim=config.GCN_HIDDEN_DIM, 
        output_dim=config.GCN_OUTPUT_DIM
    )
    H = gcn_model(X_tensor, A_tensor)
    h_CN = H[cn_idx].detach()  

    # === 处理行业死亡率表 ===
    ages_in_data = [age for age in config.AGE_RANGE if age in industry_df.index]
    years_in_data = [year for year in config.YEAR_RANGE if year in industry_df.columns]
    if not ages_in_data or not years_in_data:
        raise ValueError("行业表无可用年龄/年份交集！请检查 config 设定和数据文件！")
    df_re = industry_df.reindex(index=ages_in_data, columns=years_in_data)
    
    mask = (~df_re.isna()).astype(float).values

    mean_path = config.MEAN_CSV.format(config.FINETUNING_CORE)
    std_path = config.STD_CSV.format(config.FINETUNING_CORE)
    mean_core = pd.read_csv(mean_path, index_col=0)
    mean_core = mean_core.squeeze("columns")
    print(mean_core.index)
    std_core = pd.read_csv(std_path, index_col=0)
    std_core = std_core.squeeze("columns")
    #years_in_data = [year for year in config.YEAR_RANGE if year in df_re.columns]
    mean_arr = mean_core.loc[ages_in_data].values.reshape(-1, 1)
    std_arr = std_core.loc[ages_in_data].values.reshape(-1, 1)

    values = np.log(df_re.fillna(0).values + 1e-8)
    #values = (values - mean_arr) / std_arr

    
    if mask.sum() == 0:
        raise ValueError("行业表掩码全为0，所有数据都是 NaN 或 0，请检查 config.AGE_RANGE 和 YEAR_RANGE 设置！")
    
    num_ages, num_years = len(ages_in_data), len(years_in_data)

    # === 拼接结构特征 ===
    struct_features = h_CN.unsqueeze(0).expand(num_ages, -1)      # [num_ages, feature_dim]
    struct_features = struct_features.unsqueeze(1).expand(-1, num_years, -1) # [num_ages, num_years, feature_dim]
    mortality_seq = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)  # [num_ages, num_years, 1]
    transformer_input = torch.cat([mortality_seq, struct_features], dim=-1)  # [num_ages, num_years, 65]
    transformer_input_2d = transformer_input.reshape(1, num_ages * num_years, 65)  # [1, seq_len, 65]
    pe = get_sincos_positional_encoding(num_ages * num_years, config.POS_DIM)  # [seq_len, pos_dim]
    pe_batch = torch.tensor(pe, dtype=torch.float32).unsqueeze(0)              # [1, seq_len, pos_dim]
    transformer_input_2d = torch.cat([transformer_input_2d, pe_batch], dim=-1) # [1, seq_len, 65+pos_dim]
    mask_tensor = torch.tensor(mask, dtype=torch.float32).reshape(1, num_ages * num_years)
    target_y = torch.tensor(values, dtype=torch.float32).reshape(1, num_ages * num_years, 1)


    np.random.seed(config.VAL_SPLIT_SEED)
    num_samples = transformer_input_2d.shape[1]
    idx = np.arange(num_samples)
    np.random.shuffle(idx)
    val_size = int(num_samples * config.VAL_RATIO)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    print('mortality_seq:', mortality_seq.shape, 'min', mortality_seq.min().item(), 'max', mortality_seq.max().item())
    print('mask:', mask.shape, 'sum:', mask.sum())
    print('train_idx:', train_idx[:10], 'train count:', len(train_idx))

    return {
        'countries': ['CN'],
        'H_tensor': h_CN.unsqueeze(0),
        'mortality_data': values[np.newaxis, :, :],
        'transformer_input_2d': transformer_input_2d,
        'mask_tensor': mask_tensor,
        'target_y': target_y,
        'train_idx': train_idx,
        'val_idx': val_idx
    }
