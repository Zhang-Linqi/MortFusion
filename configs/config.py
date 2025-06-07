# -*- coding: utf-8 -*-

#==== 路径 ====
# PCA_FOLDER = your_path
# DATA_DIR = your_path
# OUTPUT_DIR = your_path
# BASE_MODEL_PATH = your_path
# MEAN_CSV = your_path
# STD_CSV = your_path
# DTW_MATRIX_PATH = your_path
#==== 模型 ====
MODEL_TYPE = 'encoderonly'   # 可选'encoderonly' 或 'seq2seq'
# ==== Decoder/MLP参数 ====
DECODER_HIDDEN_DIMS = [128, 64, 1]   # 多层线性层，每一层的维度，最后一层必须是1
DECODER_ACTIVATION = 'relu'         # 激活函数，可以是 'relu'、'gelu' 
# ==== 数据参数 ====
GENDER = 'female'
N_COMPONENTS = 3
AGE_RANGE = list(range(0, 91))               # 预测年龄
YEAR_RANGE = list(range(1950, 2019))         # 训练所用年份
FUTURE_YEARS = list(range(2019, 2120))       # 预测年份
POS_DIM = 16
FINETUNING_CORE = "HKG(Hong Kong)"
# ==== GCN参数 ====
GCN_HIDDEN_DIM = 128
GCN_OUTPUT_DIM = 64

# ==== Transformer参数 ====
TRANSFORMER_INPUT_DIM = 65 + POS_DIM
TRANSFORMER_D_MODEL = 128
TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_OUTPUT_DIM = 1
WINDOW_SIZE = 10         # 历史窗口长度
PREDICT_GAP = 1         # 预测步长
# ==== 训练/设备/随机数等 ====
DEVICE = 'cuda'  
SEED = 42

# ==== 递推控制 ====
EXPORT_HISTORY = False
EXPORT_FUTURE = True
OUTPUT_DIR = r"outputs"
TARGET_COUNTRIES = "CN"
# ==== 数据划分参数 ====
VAL_RATIO = 0.2         # 验证集比例
VAL_SPLIT_SEED = 42    
#
epochs = 20
lr = 1e-4