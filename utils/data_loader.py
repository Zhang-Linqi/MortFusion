# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from glob import glob
from typing import Tuple, List, Optional, Dict
from configs import config

class MortalityDataLoader:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir  
        self.data: Dict[str, Dict[str, pd.DataFrame]] = {"female": {}, "male": {}}
        self.stats = {"female": {}, "male": {}}  

    def load_all(self, log_transform=True, standardize=True):
        for gender in ['female', 'male']:
            folder = os.path.join(self.base_dir, gender)
            for file_path in glob(os.path.join(folder, "*.csv")):
                country_code = os.path.basename(file_path).split('_')[0]
                df = pd.read_csv(file_path, index_col=0)
                if 'Unnamed: 0' in df.columns:
                    df = df.drop(columns=['Unnamed: 0'])
                
                df.columns = [int(float(str(c))) for c in df.columns if str(c).replace('.', '', 1).isdigit()]
                year_cols = [c for c in df.columns if 1900 <= c <= 2100]
                df = df[year_cols]
                df.index = df.index.astype(int)
                df_raw = pd.read_csv(file_path)
                df.columns = df.columns.astype(int)
                df.index = df.index.astype(int)

                if log_transform:
                    #df = np.log(df + 1e-8)
                    pass
                if standardize:
                    mean = df.mean(axis=1)  
                    std = df.std(axis=1)    
                    mean = mean.reindex(df.index)
                    std = std.reindex(df.index)
                    df = df.sub(mean, axis=0).div(std, axis=0)
                    self.stats[gender][country_code] = (mean, std)
                else:
                    self.stats[gender][country_code] = (0, 1)
                self.data[gender][country_code] = df               
                print(f"正在读取文件: {file_path}")
#--------保存均值方差               
        core = config.FINETUNING_CORE
        gender = config.GENDER
        if core in self.stats[gender]:
            mean, std = self.stats[gender][core]
            mean.to_csv(config.MEAN_CSV.format(core))
            std.to_csv(config.STD_CSV.format(core))
            print(f"已导出 {core}-{gender} 的 mean/std 到 CSV")
        else:
            print(f"未找到 {core}-{gender} 数据，无法导出 mean/std")
#----------------------------------------------------------------------------------    

    def get_country_data(
        self,
        country_code: str,
        gender: str = 'female',
        age_range: Optional[list] = None,   
        year_range: Optional[list] = None
    ) -> pd.DataFrame:
        df = self.data[gender].get(country_code)
        if df is None:
            raise ValueError(f"No data found for {country_code} - {gender}")
    
        if age_range is not None:
            valid_ages = [age for age in age_range if age in df.index]
            df = df.loc[valid_ages] if valid_ages else pd.DataFrame(index=age_range, columns=df.columns)
        if year_range is not None:
            valid_years = [year for year in year_range if year in df.columns]
            df = df[valid_years] if valid_years else pd.DataFrame(columns=year_range, index=df.index)
        return df

def load_pca_features(
    self,
    folder_path: str,
    gender: str = 'female',
    n_components: int = 3
) -> Tuple[np.ndarray, List[str]]:
    X = []
    country_list = []
    gender_folder = os.path.join(folder_path, gender)
    files = sorted(glob(os.path.join(gender_folder, "*.csv")))

    for file_path in files:
        try:
            df = pd.read_csv(file_path, index_col=0)

            if df.shape[0] < n_components:
                raise ValueError(f"{file_path} 中主成分不足 {n_components} 行")

            selected = df.iloc[:n_components].values  # shape: [n_components, 年龄数]
            vec = selected.flatten()
            X.append(vec)

            country_code = os.path.basename(file_path).split("_")[0]
            country_list.append(country_code)
        except Exception as e:
            print(f" 跳过 {file_path}，错误: {e}")

    X = np.stack(X)
    return X, country_list
