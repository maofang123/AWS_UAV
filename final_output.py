import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import os
import pandas as pd
import torch
from preaatrain import TransformerModel
# torch.serialization.add_safe_globals([TransformerModel])
# 檢查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 載入前處理後的資料
data_dir = ""
X_test = np.load(os.path.join(data_dir, './processed_test_data.npy'))
# y_test = np.load(os.path.join(data_dir, '/home/r13945042/aws/testdata/y_train.npy'))

# ADD
test_data = pd.read_csv(os.path.join(data_dir, './processed_test_data.csv'))  # 假設原始測試資料在這個檔案
custumid = test_data['CUST_ID'].values  # 假設欄位名稱為 'custumid'
# 

# 轉為 tensor
X_test_tensor = torch.FloatTensor(X_test).to(device)
# y_test_tensor = torch.FloatTensor(y_test).to(device)

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

# 初始化模型
# model = TransformerModel(input_dim=X_test.shape[1], dim=64, heads=8, depth=6, dropout=0.1).to(device)

# 儲存模型 (假設模型已訓練好)
# model_path = "/home/r13945042/aws/model.pt"
# torch.save(model.state_dict(), model_path)
# print(f"模型已儲存至 {model_path}")
model_path = "./best_model/model.pt"
model = torch.load(model_path, weights_only=False)
model.to(device)
# 載入已儲存的模型
# model.load_state_dict(torch.load(model_path, weights_only=False))
# model = torch.load(model_path, weights_only=False)
model.eval()
print("模型已載入並設置為評估模式")

# ----------- 用 test set 進行 inference -----------
with torch.no_grad():
    y_proba = torch.sigmoid(model(X_test_tensor)).cpu().numpy().flatten()
# y_true = y_test
# best = {'threshold': 0.5, 'f1': 0.0}
# for t in np.linspace(0, 1, 101):
#     y_pred_t = (y_proba >= t).astype(int)
#     f1 = f1_score(y_true, y_pred_t, zero_division=0)
#     if f1 > best['f1']:
#         best = {'threshold': t, 'f1': f1}
# print(f"最佳閾值: {best['threshold']:.2f}, F1={best['f1']:.3f}")

threshold = 0.52
y_pred = (y_proba >= threshold).astype(int)

# print("Test Classification Report:")
# print(classification_report(y_true, y_pred))
# print("Test Confusion Matrix:")
# print(confusion_matrix(y_true, y_pred))

# 儲存預測結果
results_path = "./test_predictions.csv"
results = pd.DataFrame({
    # 'True_Label': y_true,
    'CUST_ID': custumid,  # 加入 custumid 欄位
    'Predicted_Label': y_pred,
    'Probability': y_proba
})
results.to_csv(results_path, index=False)
print(f"預測結果已儲存至 {results_path}")
