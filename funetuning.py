# -*- coding: utf-8 -*-


import torch
import pandas as pd
import numpy as np
import os
from configs import config
from utils.prepare_industry_inputs import prepare_industry_inputs
from train.model_runner import build_model, train_model

FUTURE_N = getattr(config, "FUTURE_YEARS", 5)  #设定预测年份
if isinstance(FUTURE_N, int):
    last_year = max(config.YEAR_RANGE)
    config.FUTURE_YEARS = [last_year + i for i in range(1, FUTURE_N + 1)]
future_years = config.FUTURE_YEARS

def main():
    # 读取行业死亡率表
    industry_df = pd.read_csv(your_path, index_col=0)
    industry_df = industry_df.applymap(lambda x: np.log(x + 1e-8)) 
    industry_df.columns = industry_df.columns.astype(int)
    data_dict = prepare_industry_inputs(config, industry_df)

    model = build_model(config, config.DEVICE)
    torch.save(model.state_dict(), config.BASE_MODEL_PATH)
    model.load_state_dict(torch.load(config.BASE_MODEL_PATH))
    for param in model.input_proj.parameters(): param.requires_grad = False
    for param in model.transformer.parameters(): param.requires_grad = False
    for param in model.output_head.parameters(): param.requires_grad = True
    train_model(config, model, data_dict, epochs=50, lr=1e-5)
    torch.save(model.state_dict(), "outputs/industry_finetuned.pth")
    print("行业表微调完成")

    model.eval()
    hist = data_dict['mortality_data'][0]   # shape: (age, year)
    num_ages, num_hist_years = hist.shape
    future_n = len(future_years)
    seq_len_total = num_hist_years + future_n
    h_CN = data_dict['H_tensor'][0].cpu().numpy()  
    TRANSFORMER_INPUT_DIM = config.TRANSFORMER_INPUT_DIM
    POS_DIM = config.POS_DIM
    pred_extended = np.zeros((num_ages, seq_len_total))
    pred_extended[:, :num_hist_years] = hist

    # 结构和位置编码拼接
    struct_feat_expanded = np.broadcast_to(h_CN.reshape(1, 1, -1), (num_ages, seq_len_total, h_CN.shape[0]))
    from utils.position_encoding import get_sincos_positional_encoding
    pe = get_sincos_positional_encoding(seq_len_total, POS_DIM)  # [seq_len_total, pos_dim]
    pe = np.broadcast_to(pe, (num_ages, seq_len_total, POS_DIM))
    struct_and_pos = np.concatenate([struct_feat_expanded, pe], axis=-1)  # [num_ages, seq_len_total, 结构+pos]

    # === 主循环：递推生成未来 N 年
    input_full = np.zeros((num_ages, seq_len_total, TRANSFORMER_INPUT_DIM))
    input_full[:, :, 1:] = struct_and_pos
    input_full[:, :num_hist_years, 0] = hist  

    for i in range(future_n):
        this_year_idx = num_hist_years + i
        flat_input = input_full[:, :this_year_idx, :].reshape(1, -1, TRANSFORMER_INPUT_DIM)
        x_pred = torch.tensor(flat_input, dtype=torch.float32).to(config.DEVICE)
        pred_seq = model(x_pred)
        pred_year = pred_seq[0, -num_ages:, 0].detach().cpu().numpy()
        input_full[:, this_year_idx, 0] = pred_year
        pred_extended[:, this_year_idx] = pred_year

    # === 导出结果
    mean_path = config.MEAN_CSV.format(config.FINETUNING_CORE)
    std_path = config.STD_CSV.format(config.FINETUNING_CORE)
    mean_core = pd.read_csv(mean_path, index_col=0).squeeze("columns")
    std_core = pd.read_csv(std_path, index_col=0).squeeze("columns")
    ages_in_data = [age for age in config.AGE_RANGE if age in industry_df.index]
    mean_arr = mean_core.loc[ages_in_data].values.reshape(-1, 1)
    std_arr = std_core.loc[ages_in_data].values.reshape(-1, 1)
    # pred_extended_restored = pred_extended * std_arr + mean_arr #反标准化可选
    pred_mortality = np.exp(pred_extended)

    all_years = list(industry_df.columns) + list(future_years)
    cols_to_use = all_years[:pred_mortality.shape[1]]
    df_pred = pd.DataFrame(pred_mortality, index=ages_in_data, columns=cols_to_use)

    os.makedirs("outputs", exist_ok=True)
    df_pred.to_csv("outputs/industry_pred_future.csv")
    print(f"递归预测并保存成功：outputs/industry_pred_future.csv")

if __name__ == "__main__":
    main()

