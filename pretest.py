import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 載入資料
id_df = pd.read_csv("./比賽用資料/Test/(Test)ID_Data_202501.csv")
accts_df = pd.read_csv("./比賽用資料/Test/(Test)ACCTS_Data_202501.csv")
txn_df = pd.read_csv("./比賽用資料/Test/(Test)SAV_TXN_Data_202501.csv")

# id_df = pd.read_csv("/home/r13945042/aws/比賽用資料/Train/(Train)ID_Data_202412.csv")
# accts_df = pd.read_csv("/home/r13945042/aws/比賽用資料/Train/(Train)ACCTS_Data_202412.csv")
# # eccus_df = pd.read_csv("/home/r13945042/aws/比賽用資料/Train/(Train)ECCUS_Data_202412.csv")
# txn_df = pd.read_csv("/home/r13945042/aws/比賽用資料/Train/(Train)SAV_TXN_Data_202412.csv")

# 資料清理
id_df['YEARLYINCOMELEVEL'] = id_df['YEARLYINCOMELEVEL'].fillna(id_df['YEARLYINCOMELEVEL'].median())
id_df['CNTY_CD'] = id_df['CNTY_CD'].fillna(id_df['CNTY_CD'].mode()[0])
txn_df = txn_df.fillna(0)

# 補AGE欄位
id_df['AGE'] = id_df['DATE_OF_BIRTH']

# 收入水平分類
id_df['INCOME_CATEGORY'] = pd.cut(id_df['YEARLYINCOMELEVEL'],
                                  bins=[0, 30, 60, 100, 200, 500, float('inf')],
                                  labels=['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Ultra High'])

# 統計每個客戶的帳戶數
account_counts = accts_df.groupby('CUST_ID')['ACCT_NBR'].count().reset_index()
account_counts.rename(columns={'ACCT_NBR': 'NUM_ACCOUNTS'}, inplace=True)

# 交易資料特徵提取
txn_frequency = txn_df.groupby('CUST_ID').size().reset_index(name='TXN_FREQUENCY')

txn_amount_stats = txn_df.groupby('CUST_ID')['TX_AMT'].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
txn_amount_stats.columns = ['CUST_ID', 'TOTAL_TXN_AMOUNT', 'AVG_TXN_AMOUNT', 'STD_TXN_AMOUNT', 'MIN_TXN_AMOUNT', 'MAX_TXN_AMOUNT']

channel_counts = pd.crosstab(txn_df['CUST_ID'], txn_df['CHANNEL_CODE'])
channel_counts = channel_counts.div(channel_counts.sum(axis=1), axis=0)
channel_counts = channel_counts.add_prefix('CHANNEL_RATIO_')

txn_df['TX_HOUR'] = txn_df['TX_TIME']
txn_df['UNUSUAL_HOUR'] = ((txn_df['TX_HOUR'] < 8) | (txn_df['TX_HOUR'] > 18)).astype(int)
unusual_hour_ratio = txn_df.groupby('CUST_ID')['UNUSUAL_HOUR'].mean().reset_index(name='UNUSUAL_HOUR_RATIO')

ip_uuid_features = txn_df.groupby('CUST_ID')[['SAME_NUMBER_IP', 'SAME_NUMBER_UUID']].mean().reset_index()
ip_uuid_features.columns = ['CUST_ID', 'AVG_SAME_IP_RATIO', 'AVG_SAME_UUID_RATIO']

# 合併特徵
features = id_df[['CUST_ID', 'AUM_AMT', 'AGE', 'INCOME_CATEGORY', 'CNTY_CD']]
features = pd.merge(features, account_counts, on='CUST_ID', how='left')
features = pd.merge(features, txn_frequency, on='CUST_ID', how='left')
features = pd.merge(features, txn_amount_stats, on='CUST_ID', how='left')
features = pd.merge(features, channel_counts, on='CUST_ID', how='left')
features = pd.merge(features, unusual_hour_ratio, on='CUST_ID', how='left')
features = pd.merge(features, ip_uuid_features, on='CUST_ID', how='left')

# 修正缺失值處理
features['INCOME_CATEGORY'] = features['INCOME_CATEGORY'].cat.add_categories(['Unknown'])
features['INCOME_CATEGORY'] = features['INCOME_CATEGORY'].fillna('Unknown')
numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
features[numeric_cols] = features[numeric_cols].fillna(0)

# 建立 target 標籤 (測試集不需要，這裡僅保留 CUST_ID)
target = pd.DataFrame({
    'CUST_ID': id_df['CUST_ID'],
})

# 合併特徵和目標
model_data = pd.merge(features, target, on='CUST_ID', how='left')

# 準備特徵變數
X = model_data.drop(['CUST_ID'], axis=1)

# 這是你指定要的特徵（注意，是原始欄位名）
selected_features_real_names = [
    'AUM_AMT', 'AGE', 'TXN_FREQUENCY', 'TOTAL_TXN_AMOUNT', 'AVG_TXN_AMOUNT',
    'STD_TXN_AMOUNT', 'MIN_TXN_AMOUNT', 'MAX_TXN_AMOUNT',
    'CHANNEL_RATIO_10', 'CHANNEL_RATIO_14', 'CHANNEL_RATIO_15',
    'CHANNEL_RATIO_16', 'CHANNEL_RATIO_17', 'CHANNEL_RATIO_18', 'CHANNEL_RATIO_19',
    'UNUSUAL_HOUR_RATIO', 'AVG_SAME_IP_RATIO', 'AVG_SAME_UUID_RATIO'
]

# 特徵選擇
X_selected = X[selected_features_real_names]

# 標準化數值特徵
scaler = StandardScaler()
X_selected_scaled = scaler.fit_transform(X_selected)

# 將標準化後的資料轉為 numpy array
X_output = X_selected_scaled

# 儲存為 .npy 檔案
np.save('./processed_test_data.npy', X_output)

print("✅ 成功將資料輸出為 processed_test_data.npy")
