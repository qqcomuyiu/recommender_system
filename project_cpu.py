import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import optuna
from tqdm import tqdm
import dataop
import layers
class gpu_v:
    def run():
# 强制使用 CPU
        torch.cuda.is_available = lambda: False
        device = torch.device("cpu")
        
        # 加载数据
        print("Loading data ...")
        items1 = pd.read_csv('C:\\tensor\\item_properties_part1.csv')
        items2 = pd.read_csv('C:\\tensor\\item_properties_part2.csv')
        events = pd.read_csv('C:\\tensor\\events.csv')
        category_tree = pd.read_csv('C:\\tensor\\category_tree.csv')
        
        # 数据预处理
        #events = events.head(2000)
        grouped = events.groupby('event')['itemid'].apply(list)
        sequences_tensor = dataop.data_process(grouped)
        sequences_numpy = sequences_tensor.numpy()
        
        # 第一次划分：70% 训练集，30% 测试 + 验证集
        X_train, X_temp = train_test_split(sequences_numpy, test_size=0.3, random_state=42)
        
        # 第二次划分：30% 中的 50% 用作验证集，50% 用作测试集
        X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)
        
        # 检查划分后的形状
        print(f"训练集大小: {X_train.shape}")
        print(f"验证集大小: {X_val.shape}")
        print(f"测试集大小: {X_test.shape}")
        
        # 输入维度设置
        input_dim = X_train.shape[1]  # 输入维度，等于特征数（列数）
        maxlen = X_train.shape[0]     # 最大序列长度，等于训练集的样本数
        
        def objective(trial):
            # 定义要优化的超参数
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            num_heads = trial.suggest_categorical("num_heads", [1, 4, 8, 16])
            ff_dim = trial.suggest_int("ff_dim", 64, 512, step=64)
            num_encoders = trial.suggest_int("num_encoders", 1, 6)
            num_decoders = trial.suggest_int("num_decoders", 1, 6)
            num_layers = trial.suggest_int("num_layers", 1, 4)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            embed_dim = trial.suggest_int("embed_dim", 64, 256, step=64)
            #batch_size = trial.suggest_int("batch_size", 16, 128,step = 16)
            positive_weight = trial.suggest_int("positive_weight", 1, 6)
            # 初始化模型
            model = layers.TransformerRecommenderWithMixedAttention(
                input_dim=input_dim,
                maxlen=maxlen,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_encoders=num_encoders,
                num_decoders=num_decoders,
                num_layers=num_layers,
                dropout=dropout,
            ).to(device)
        
            # 使用 AdamW 作为优化器
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        
            # 使用 ReduceLROnPlateau 作为学习率调度器
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
        
            # 损失函数：加权二元交叉熵损失
            #positive_weight = 3.0  # 正样本权重
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight]).to(device))
        
            # 训练参数
            epochs = 50
            batch_size = 128
            early_stop_patience = 7  # 早停法 patience 设置为 7 个 epoch
            best_val_loss = float('inf')
            early_stop_counter = 0
        
            for epoch in tqdm(range(epochs), desc="训练进度", leave=False):
                model.train()
                total_loss = 0
                total_correct = 0  # 累计正确预测的数量
                total_samples = 0  # 累计样本数量
        
                # 添加进度条到批次循环
                with tqdm(total=len(X_train), desc=f"Epoch {epoch+1}/{epochs}", leave=False) as pbar:
                    for i in range(0, len(X_train), batch_size):
                        inputs = torch.tensor(X_train[i:i + batch_size]).to(device)
                        inputs = torch.clamp(inputs, 0, input_dim - 1)
                        targets = (inputs > 0).float().to(device)  # 将目标值转换为二元（0/1）
        
                        optimizer.zero_grad()
                        outputs = model(inputs, inputs)  # 输出 logits，尚未应用 sigmoid
        
                        # 计算二元交叉熵加权损失
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
        
                        total_loss += loss.item() * inputs.size(0)
        
                        # 计算准确率
                        probs = torch.sigmoid(outputs)
                        preds = (probs >= 0.5).float()
                        correct = (preds == targets).sum().item()
                        total_correct += correct
                        total_samples += targets.numel()
        
                        pbar.update(inputs.size(0))  # 更新进度条
        
                avg_train_loss = total_loss / len(X_train)
                train_accuracy = total_correct / total_samples
        
                # 验证集
                model.eval()
                with torch.no_grad():
                    val_inputs = torch.tensor(X_val).to(device)
                    val_inputs = torch.clamp(val_inputs, 0, input_dim - 1)
                    val_targets = (val_inputs > 0).float().to(device)  # 将目标转换为二元
                    val_outputs = model(val_inputs, val_inputs)
                    val_loss = criterion(val_outputs, val_targets)
        
                    # 计算验证集准确率
                    val_probs = torch.sigmoid(val_outputs)
                    val_preds = (val_probs >= 0.5).float()
                    val_correct = (val_preds == val_targets).sum().item()
                    val_total = val_targets.numel()
                    val_accuracy = val_correct / val_total
        
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}")
        
                # 调整学习率
                scheduler.step(val_loss.item())
        
                # 早停法：如果验证损失没有改善则计数增加
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
        
                if early_stop_counter >= early_stop_patience:
                    print("Early stopping triggered.")
                    break
        
            return best_val_loss
        
        # 使用 tqdm 显示优化进度
        n_trials = 5
        with tqdm(total=n_trials, desc="超参数优化进度") as pbar:
            def wrapped_objective(trial):
                val_loss = objective(trial)
                pbar.update(1)
                return val_loss
        
            study = optuna.create_study(direction="minimize")
            study.optimize(wrapped_objective, n_trials=n_trials)
        
        # 打印最佳的超参数
        print(f"Best trial: {study.best_trial.params}")
