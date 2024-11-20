import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import optuna
from tqdm import tqdm
import dataop
import layers
import random
# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据
items1 = pd.read_csv('C:\\tensor\\item_properties_part1.csv')
items2 = pd.read_csv('C:\\tensor\\item_properties_part2.csv')
events = pd.read_csv('C:\\tensor\\events.csv')
category_tree = pd.read_csv('C:\\tensor\\category_tree.csv')

# 使用 LabelEncoder 对 itemid 进行编码
le = LabelEncoder()
events['itemid'] = le.fit_transform(events['itemid'])

# 获取词汇表大小
vocab_size = len(le.classes_)

# 处理数据
# 假设 dataop.data_process 函数需要修改为适应编码后的 itemid
grouped = events.groupby('event')['itemid'].apply(list)

# 将序列转换为张量列表，并进行填充
# 检查数据中是否存在 session_id 或 user_id 列
if 'session_id' in events.columns:
    # 按 session_id 分组
    grouped = events.groupby('session_id')['itemid'].apply(list)
elif 'user_id' in events.columns:
    # 按 user_id 分组
    grouped = events.groupby('user_id')['itemid'].apply(list)
elif 'visitorid' in events.columns:
    # 按 user_id 分组
    grouped = events.groupby('visitorid')['itemid'].apply(list)
else:
    # 如果都没有，可能需要根据其他列或自定义方式进行分组
    raise ValueError("No suitable column found for grouping the data.")

sequences = grouped.tolist()
sample_size = 100000  # 采样10万条数据
if len(sequences) > sample_size:
    sequences = random.sample(sequences, sample_size)
print(f"Number of sequences before padding: {len(sequences)}")
sequences_padded = nn.utils.rnn.pad_sequence(
    [torch.tensor(seq) for seq in sequences],
    batch_first=True,
    padding_value=0  # 使用索引 0 进行填充
)
sequences_numpy = sequences_padded.numpy()
print(f"Shape of sequences_padded: {sequences_padded.shape}")
print(f"Shape of sequences_numpy: {sequences_numpy.shape}")
# 数据集划分
X_train, X_temp = train_test_split(sequences_numpy, test_size=0.3, random_state=42)

# 第二次划分：30% 中的 50% 用作验证集，50% 用作测试集
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

# 检查划分后的形状
print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_val.shape}")
print(f"测试集大小: {X_test.shape}")

# 模型参数
input_dim = vocab_size + 1  # 加1，因为我们使用了填充索引0
maxlen = X_train.shape[1]  # 序列长度
embed_dim = 128
num_heads = 8
ff_dim = 128
num_encoders = 2
num_decoders = 2
num_layers = 2
dropout = 0.1

# 初始化模型并移动到 GPU
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

# 检查模型输出
batch_size = 32
sequence_length = maxlen  # 使用实际的序列长度
dummy_input = torch.randint(0, input_dim, (batch_size, sequence_length), dtype=torch.long).to(device)
dummy_output = model(dummy_input, dummy_input)
print("模型输出维度:", dummy_output.shape)

# Objective 函数
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))

    epochs = 50
    batch_size = 32
    early_stop_patience = 7
    best_val_loss = float('inf')
    early_stop_counter = 0

    # 将数据移动到 GPU
    train_tensor = torch.tensor(X_train, dtype=torch.long).to(device)
    val_tensor = torch.tensor(X_val, dtype=torch.long).to(device)
    test_tensor = torch.tensor(X_test, dtype=torch.long).to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(train_tensor), batch_size):
            inputs = train_tensor[i:i + batch_size]
            targets = (inputs > 0).float()  # 转换为二元目标
            optimizer.zero_grad()
            outputs = model(inputs, inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        avg_train_loss = total_loss / len(train_tensor)

        # 验证集
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor, val_tensor)
            val_targets = (val_tensor > 0).float()
            val_loss = criterion(val_outputs, val_targets)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}')

        scheduler.step(val_loss.item())

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

    return best_val_loss

# 使用 tqdm 显示进度条
n_trials = 50
with tqdm(total=n_trials, desc="Optimizing", dynamic_ncols=True) as pbar:
    def wrapped_objective(trial):
        val_loss = objective(trial)
        pbar.set_postfix({"Best Loss": study.best_value if study.best_value else "N/A", "Current Loss": val_loss})
        pbar.update(1)
        return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(wrapped_objective, n_trials=n_trials)

# 打印最佳超参数
print(f"Best trial: {study.best_trial.params}")
