# -*- coding: utf-8 -*-

import os
import pandas as pd
from dtaidistance import dtw
from itertools import combinations

#folder = your_path

dfs = {}
for filename in os.listdir(folder):
    if filename.endswith('.csv'):
        country = filename.split('_')[0]
        df = pd.read_csv(os.path.join(folder, filename), index_col=0)
        df.index = df.index.astype(str)
        dfs[country] = df

countries = list(dfs.keys())
dtw_matrix = pd.DataFrame(index=countries, columns=countries, dtype=float)

for c1, c2 in combinations(countries, 2):
    df1, df2 = dfs[c1], dfs[c2]
    ages = df1.index.intersection(df2.index)
    arr1 = df1.loc[ages].values.flatten()
    arr2 = df2.loc[ages].values.flatten()
    minlen = min(len(arr1), len(arr2))
    arr1, arr2 = arr1[:minlen], arr2[:minlen]
    mask = ~pd.isnull(arr1) & ~pd.isnull(arr2)
    arr1, arr2 = arr1[mask], arr2[mask]
    distance = dtw.distance(arr1, arr2)
    dtw_matrix.loc[c1, c2] = distance
    dtw_matrix.loc[c2, c1] = distance

for c in countries:
    dtw_matrix.loc[c, c] = 0

dtw_matrix.to_csv('dtw_distance_matrix_no_year_alignment_female.csv')
print(dtw_matrix)