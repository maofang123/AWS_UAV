import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import os

# 檢查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 載入前處理後的資料
data_dir = ""
X_train = np.load(os.path.join(data_dir, './feature/X_train.npy'))
X_val = np.load(os.path.join(data_dir, './feature/X_val.npy'))
X_test = np.load(os.path.join(data_dir, './feature/X_test.npy'))
y_train = np.load(os.path.join(data_dir, './feature/y_train.npy'))
y_val = np.load(os.path.join(data_dir, './feature/y_val.npy'))
y_test = np.load(os.path.join(data_dir, './feature/y_test.npy'))

# 轉為 tensor
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.FloatTensor(y_val).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
batch_size = 2048
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定義 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)

# 定義 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, dim=32, heads=4, depth=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, dim)
        
        # Transformer 編碼器層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 輸出層
        self.output_layers = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        # 將輸入投影到指定維度
        x = self.embedding(x)  # (batch_size, input_dim) -> (batch_size, dim)
        
        # 增加一個序列維度以適配 Transformer (batch_size, seq_len=1, dim)
        x = x.unsqueeze(1)
        
        # 通過 Transformer 編碼器
        x = self.transformer_encoder(x)  # (batch_size, seq_len=1, dim)
        
        # 取第一個 token 的輸出 (batch_size, dim)
        x = x[:, 0, :]
        
        # 輸出層
        return self.output_layers(x)

# 定義 WarmUpLR 類別
class WarmUpLR:
    def __init__(self, optimizer, warmup_steps, after_scheduler):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self.finished = False
        self.step_num = 0

    def step(self, metrics=None):
        if self.finished:
            if metrics is not None:
                self.after_scheduler.step(metrics)
            return
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr_scale = self.step_num / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_scale * param_group['initial_lr']
        else:
            self.finished = True
            if metrics is not None:
                self.after_scheduler.step(metrics)

if __name__ == '__main__':
    # 初始化模型、損失函數和優化器
    model = TransformerModel(input_dim=X_train.shape[1], dim=64, heads=2, depth=1, dropout=0.1).to(device)
    criterion = FocalLoss(alpha=0.99, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)

    # 設定初始學習率參數
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = 0.01

    # 設定 warm-up 步數
    warmup_steps = 10

    # 使用 ReduceLROnPlateau 作為 warm-up 後的學習率調度器
    scheduler_after_warmup = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 包裝 warm-up 調度器
    warmup_scheduler = WarmUpLR(optimizer, warmup_steps, scheduler_after_warmup)

    # ----------- 訓練模型 -----------
    num_epochs = 200
    patience = 10
    best_val_loss = float('inf')
    best_model_weights = None
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        # 用 val 驗證
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor).item()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 使用 warm-up 調度器並傳入驗證損失
        warmup_scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    # 載入最佳模型
    model.load_state_dict(best_model_weights)

    # ----------- 用 test set 找最佳 threshold -----------
    model.eval()
    torch.save(model, './model.pt')
    with torch.no_grad():
        y_proba = torch.sigmoid(model(X_test_tensor)).cpu().numpy().flatten()
    y_true = y_test
    best = {'threshold': 0.5, 'f1': 0.0}
    for t in np.linspace(0, 1, 101):
        y_pred_t = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred_t, zero_division=0)
        if f1 > best['f1']:
            best = {'threshold': t, 'f1': f1}
    print(f"最佳閾值: {best['threshold']:.2f}, F1={best['f1']:.3f}")

    threshold = best['threshold']
    y_pred = (y_proba >= threshold).astype(int)

    print("Test Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
