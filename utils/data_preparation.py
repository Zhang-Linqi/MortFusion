# -*- coding: utf-8 -*-


import torch
import numpy as np
import pandas as pd
from models.mortality_gcn import build_input_matrix_from_ready_features, GCNEncoder
from utils.data_loader import MortalityDataLoader
from configs import config
def get_sincos_positional_encoding(num_positions, d_model):
    position = np.arange(num_positions)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((num_positions, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(pe, dtype=torch.float32)  

def prepare_model_inputs(config):
    X_np, countries = build_input_matrix_from_ready_features(
        folder_path=config.PCA_FOLDER,
        gender=config.GENDER,
        n_components=config.N_COMPONENTS
    )
    X_tensor = torch.tensor(X_np, dtype=torch.float32)

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
    
    loader = MortalityDataLoader(config.DATA_DIR)
    loader.load_all()
    age_range = config.AGE_RANGE
    year_range = config.YEAR_RANGE
    mortality_data_list, mask_list = [], []

    for country_code in countries:
        try:
            df = loader.get_country_data(country_code, gender=config.GENDER)
            df = df[[c for c in df.columns if isinstance(c, int) and 1900 <= c <= 2100]]
            df_re = df.reindex(index=age_range, columns=year_range)
            mask = (~df_re.isna()).astype(float).values
            values = df_re.fillna(0).values
            mortality_data_list.append(values)
            mask_list.append(mask)
    
        except Exception as e:
            print(f"国家 {country_code} 读取异常：{e}")
    mortality_data = np.stack(mortality_data_list)
    mask_data = np.stack(mask_list)

    num_ages, num_years = len(age_range), len(year_range)
    H_tensor = torch.tensor(H, dtype=torch.float32)
    struct_features = H_tensor.unsqueeze(1).expand(-1, num_ages, -1)
    struct_features = struct_features.unsqueeze(2).expand(-1, -1, num_years, -1)
    mortality_seq = torch.tensor(mortality_data, dtype=torch.float32).unsqueeze(-1)
    transformer_input = torch.cat([mortality_seq, struct_features], dim=-1)
    transformer_input_2d = transformer_input.reshape(len(countries), num_ages * num_years, 65) 

    num_positions = transformer_input_2d.shape[1]

    pe = get_sincos_positional_encoding(num_ages * num_years, config.POS_DIM)  # [seq_len, pos_dim]
    pe_batch = pe.unsqueeze(0).expand(transformer_input_2d.shape[0], -1, -1)   # [batch, seq_len, pos_dim]
    

    transformer_input_2d = torch.cat([transformer_input_2d, pe_batch], dim=-1) # [batch, seq_len, 65+POS_DIM]transformer_input_2d = torch.cat([transformer_input_2d, pe_batch], dim=-1)  # 变成[batch, seq, 65+POS_DIM]
    mask_tensor = torch.tensor(mask_data, dtype=torch.float32).reshape(len(countries), num_ages * num_years)
    target_y = torch.tensor(mortality_data, dtype=torch.float32).reshape(len(countries), num_ages * num_years, 1)


    np.random.seed(config.VAL_SPLIT_SEED)
    num_samples = transformer_input_2d.shape[1] 
    idx = np.arange(num_samples)
    np.random.shuffle(idx)
    val_size = int(num_samples * config.VAL_RATIO)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    return {
        'countries': countries,
        'H_tensor': H_tensor,
        'mortality_data': mortality_data,
        'transformer_input_2d': transformer_input_2d,
        'mask_tensor': mask_tensor,
        'target_y': target_y,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'loader': loader
    }

def prepare_seq2seq_inputs(config):

    base = prepare_model_inputs(config)
    countries = base['countries']
    transformer_input_2d = base['transformer_input_2d']  # [num_countries, seq_len, dim]
    target_y = base['target_y']  # [num_countries, seq_len, 1]
    num_countries = len(countries)
    num_ages = len(config.AGE_RANGE)
    num_years = len(config.YEAR_RANGE)
    num_future = len(config.FUTURE_YEARS)
    input_dim = transformer_input_2d.shape[-1]

    src_inputs = []
    tgt_inputs = []
    tgt_ys = []

    for i in range(num_countries):
        for age_idx in range(num_ages):
            hist_start = age_idx * num_years
            hist_end = (age_idx + 1) * num_years
            
            src = transformer_input_2d[i, hist_start:hist_end, :]  


            first_dec = src[-1:].clone()  
            future_zeros = torch.zeros((num_future - 1, input_dim), dtype=src.dtype)
            tgt_in = torch.cat([first_dec, future_zeros], dim=0)  


            y_true = target_y[i, hist_end:hist_end + num_future, :]  

            src_inputs.append(src)
            tgt_inputs.append(tgt_in)
            tgt_ys.append(y_true)

    src_input_tensor = torch.stack(src_inputs)  
    tgt_input_tensor = torch.stack(tgt_inputs)  
    tgt_y_tensor = torch.stack(tgt_ys)          

    data_dict = {
        'countries': countries,
        'H_tensor': base['H_tensor'],
        'mortality_data': base['mortality_data'],
        'src_input': src_input_tensor,
        'tgt_input': tgt_input_tensor,
        'tgt_y': tgt_y_tensor,
        'train_idx': base['train_idx'],
        'val_idx': base['val_idx'],
        'loader': base['loader']
    }
    return data_dict


def prepare_inputs(config):
    if getattr(config, "MODEL_TYPE", "encoderonly") == "encoderonly":
        return prepare_model_inputs(config)
    elif config.MODEL_TYPE == "seq2seq":
        return prepare_seq2seq_inputs(config)
    else:
        raise ValueError(f"不支持的模型类型: {config.MODEL_TYPE}")