import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import matplotlib.font_manager as fm

# 載入資料 (改成Colab上傳後的路徑)
id_df = pd.read_csv("./比賽用資料/Train/(Train)ID_Data_202412.csv")
accts_df = pd.read_csv("./比賽用資料/Train/(Train)ACCTS_Data_202412.csv")
eccus_df = pd.read_csv("./比賽用資料/Train/(Train)ECCUS_Data_202412.csv")
txn_df = pd.read_csv("./比賽用資料/Train/(Train)SAV_TXN_Data_202412.csv")

# 資料清理
id_df['YEARLYINCOMELEVEL'] = id_df['YEARLYINCOMELEVEL'].fillna(id_df['YEARLYINCOMELEVEL'].median())
id_df['CNTY_CD'] = id_df['CNTY_CD'].fillna(id_df['CNTY_CD'].mode()[0])


# 處理缺失值
id_df['YEARLYINCOMELEVEL'] = id_df['YEARLYINCOMELEVEL'].fillna(id_df['YEARLYINCOMELEVEL'].median())
id_df['CNTY_CD'] = id_df['CNTY_CD'].fillna(id_df['CNTY_CD'].mode()[0])

# 處理交易資料缺失值
txn_df = txn_df.fillna(0)
id_df['AGE'] = id_df['DATE_OF_BIRTH']
# 收入水平分類
id_df['INCOME_CATEGORY'] = pd.cut(id_df['YEARLYINCOMELEVEL'],
                                  bins=[0, 30, 60, 100, 200, 500, float('inf')],
                                  labels=['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Ultra High'])

# 統計每個客戶的帳戶數
account_counts = accts_df.groupby('CUST_ID')['ACCT_NBR'].count().reset_index()
account_counts.rename(columns={'ACCT_NBR': 'NUM_ACCOUNTS'}, inplace=True)

# 收入水平分類
id_df['INCOME_CATEGORY'] = pd.cut(id_df['YEARLYINCOMELEVEL'],
                                  bins=[0, 30, 60, 100, 200, 500, float('inf')],
                                  labels=['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Ultra High'])

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

# 創建目標標籤（是否為警示戶）
target = pd.DataFrame({
    'CUST_ID': id_df['CUST_ID'],
    'IS_ALERT': id_df['CUST_ID'].isin(eccus_df['CUST_ID']).astype(int)
})

# 合併特徵和目標
model_data = pd.merge(features, target, on='CUST_ID', how='left')

# 準備特徵和目標變數
X = model_data.drop(['CUST_ID', 'IS_ALERT'], axis=1)
y = model_data['IS_ALERT']

# 特徵選擇
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

# 使用隨機森林進行特徵選擇，並記錄被選中的特徵
selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
selector.fit(X_processed, y)
X_selected = selector.transform(X_processed)

# 處理資料不平衡
sampler = Pipeline([
    ('over', SMOTE(sampling_strategy=0.1)),
    ('under', RandomUnderSampler(sampling_strategy=0.1))
])

X_resampled, y_resampled = sampler.fit_resample(X_selected, y)

# 分割資料為訓練、測試和驗證集
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

output_dir = "./feature"

# 儲存處理後的資料
np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

# np.save(os.path.join(output_dir, 'X_train.npy'), X_selected)
# np.save(os.path.join(output_dir, 'y_train.npy'), y)




