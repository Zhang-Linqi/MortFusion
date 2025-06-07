# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.predictor import recursive_predict_and_export
from configs import config

def build_model(config, device):
    """
    根据 config.MODEL_TYPE 构建 Encoder-only 或 Seq2Seq 模型
    """
    if getattr(config, "MODEL_TYPE", "encoderonly") == "encoderonly":
        from models.transformer_encoderonly import MortalityTransformer
        return MortalityTransformer(
            input_dim=config.TRANSFORMER_INPUT_DIM,
            d_model=config.TRANSFORMER_D_MODEL,
            nhead=config.TRANSFORMER_NHEAD,
            num_layers=config.TRANSFORMER_NUM_LAYERS,
            output_dim=config.TRANSFORMER_OUTPUT_DIM
        ).to(device)
    elif config.MODEL_TYPE == "seq2seq":
        from models.transformer_seq2seq import MortalityTransformerSeq2Seq
        return MortalityTransformerSeq2Seq(
            input_dim=config.TRANSFORMER_INPUT_DIM,
            d_model=config.TRANSFORMER_D_MODEL,
            nhead=config.TRANSFORMER_NHEAD,
            num_layers=config.TRANSFORMER_NUM_LAYERS,
            output_dim=config.TRANSFORMER_OUTPUT_DIM,
            max_seq_len=getattr(config, "MAX_SEQ_LEN", 2048)
        ).to(device)
    else:
        raise ValueError(f"不支持的模型类型: {config.MODEL_TYPE}")

def run_history_prediction(config, model, data_dict):
    countries = data_dict['countries']
    device = next(model.parameters()).device

    all_preds, all_losses = [], []
    if config.MODEL_TYPE == "encoderonly":
        transformer_input_2d = data_dict['transformer_input_2d']
        mask_tensor = data_dict['mask_tensor']
        target_y = data_dict['target_y']
        for i in tqdm(range(len(countries)), desc="国家历史期批量预测"):
            x = transformer_input_2d[i:i+1].to(device)
            y = target_y[i:i+1].to(device)
            mask = mask_tensor[i:i+1].to(device)
            pred = model(x)
            from models.transformer_encoderonly import masked_mse_loss
            loss = masked_mse_loss(pred, y, mask)
            all_preds.append(pred.detach().cpu().numpy())
            all_losses.append(loss.item())
    elif config.MODEL_TYPE == "seq2seq":
        src_input = data_dict['src_input']
        tgt_input = data_dict['tgt_input']
        tgt_y = data_dict['tgt_y']
        for i in tqdm(range(len(countries)), desc="国家历史期批量预测"):
            x_src = src_input[i:i+1].to(device)
            x_tgt = tgt_input[i:i+1].to(device)
            y = tgt_y[i:i+1].to(device)
            pred = model(x_src, x_tgt)
            # 这里假设用MSE即可
            loss = torch.mean((pred - y) ** 2)
            all_preds.append(pred.detach().cpu().numpy())
            all_losses.append(loss.item())
    print("✅ 历史期预测全部完成！")
    return all_preds, all_losses

def run_recursive_prediction(config, model, data_dict):
    out = recursive_predict_and_export(
        model=model,
        countries=data_dict['countries'],
        H_tensor=data_dict['H_tensor'],
        mortality_data=data_dict['mortality_data'],
        OUTPUT_DIR=config.OUTPUT_DIR,
        age_range=config.AGE_RANGE,
        year_range=config.YEAR_RANGE,
        future_years=config.FUTURE_YEARS,
        TRANSFORMER_INPUT_DIM=config.TRANSFORMER_INPUT_DIM,
        model_type=getattr(config, "MODEL_TYPE", "encoderonly"),
        target_countries=getattr(config, "TARGET_COUNTRIES", None),
        loader=data_dict['loader'] 
    )
    return out

def train_one_epoch(model, optimizer, config, data_dict):
    model.train()
    train_idx = data_dict['train_idx']
    total_loss = 0
    n = 0
    if config.MODEL_TYPE == "encoderonly":
        for i in range(len(data_dict['countries'])):
            x = data_dict['transformer_input_2d'][i:i+1, train_idx].to(config.DEVICE)
            y = data_dict['target_y'][i:i+1, train_idx].to(config.DEVICE)
            mask = data_dict['mask_tensor'][i:i+1, train_idx].to(config.DEVICE)
            pred = model(x)
            from models.transformer_encoderonly import masked_mse_loss
            loss = masked_mse_loss(pred, y, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
            print("mask nonzero:", mask.sum().item())
            print("pred:", pred[:5])
            print("y:", y[:5])
            print("loss:", loss.item())
            
    elif config.MODEL_TYPE == "seq2seq":
        for i in range(len(data_dict['countries'])):
            x_src = data_dict['src_input'][i:i+1].to(config.DEVICE)
            x_tgt = data_dict['tgt_input'][i:i+1].to(config.DEVICE)
            y = data_dict['tgt_y'][i:i+1].to(config.DEVICE)  
            pred = model(x_src, x_tgt)
            assert pred.shape == y.shape, f"pred: {pred.shape}, y: {y.shape}"
            loss = torch.mean((pred - y) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        else:
            raise ValueError(f"不支持的模型类型: {config.MODEL_TYPE}")
    return total_loss / n

def evaluate(model, config, data_dict, split='val'):
    model.eval()
    idx = data_dict['val_idx'] if split == 'val' else data_dict['train_idx']
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        if config.MODEL_TYPE == "encoderonly":
            for i in range(len(data_dict['countries'])):
                x = data_dict['transformer_input_2d'][i:i+1, idx].to(config.DEVICE)
                y = data_dict['target_y'][i:i+1, idx].to(config.DEVICE)
                mask = data_dict['mask_tensor'][i:i+1, idx].to(config.DEVICE)
                pred = model(x)
                from models.transformer_encoderonly import masked_mse_loss
                loss = masked_mse_loss(pred, y, mask)
                pred_np = pred.cpu().numpy().flatten()
                y_np = y.cpu().numpy().flatten()
                mask_np = mask.cpu().numpy().flatten()
                valid = mask_np > 0
                all_preds.append(pred_np[valid])
                all_targets.append(y_np[valid])
                total_loss += loss.item()
        elif config.MODEL_TYPE == "seq2seq":
            for i in range(len(data_dict['countries'])):
                x_src = data_dict['src_input'][i:i+1].to(config.DEVICE)
                x_tgt = data_dict['tgt_input'][i:i+1].to(config.DEVICE)
                y = data_dict['tgt_y'][i:i+1].to(config.DEVICE)
                pred = model(x_src, x_tgt)
                loss = torch.mean((pred - y) ** 2)
                all_preds.append(pred.cpu().numpy().flatten())
                all_targets.append(y.cpu().numpy().flatten())
                total_loss += loss.item()
        else:
            raise ValueError(f"不支持的模型类型: {config.MODEL_TYPE}")
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    return total_loss / len(data_dict['countries']), rmse

def train_model(config, model, data_dict, epochs=None, lr=None):
    if epochs is None:
        epochs = config.epochs
    if lr is None:
        lr = config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    
    train_loss_list = []
    val_loss_list = []
    val_rmse_list = []
    
    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, optimizer, config, data_dict)
        val_loss, val_rmse = evaluate(model, config, data_dict, split='val')
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_rmse_list.append(val_rmse)

        print(f"Epoch {epoch:02d} | TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | ValRMSE: {val_rmse:.4f}")
    print("训练完成！")
    torch.save(model.state_dict(), "outputs/base_model.pth")
    print("模型权重已保存为 outputs/base_model.pth")
    # ===   loss 曲线 ===
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/loss_curve.png")
    plt.show()
    
    return train_loss_list, val_loss_list, val_rmse_list