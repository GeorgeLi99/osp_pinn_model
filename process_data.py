import pandas as pd
import numpy as np
from config import *

path=DATA_PATH
f=open(path,'r',encoding='utf-8')
# 修改：使用header=0表示第一行是标题行
data=pd.read_csv(f,header=0)
record_num = len(data)
print(f'number of records: {record_num}')

# 打印数据类型信息以便调试
print("数据类型信息:")
print(data.dtypes)
print("\n数据预览:")
print(data.head())

# 提取特征和标签，并明确转换为浮点数类型
try:
    # 尝试使用列索引
    features_df=data.iloc[:,0:COLUMN]
    labels_df=data.iloc[:,COLUMN]
except IndexError:
    # 如果索引越界，打印列名并尝试使用列名
    print(f"CSV列名: {data.columns.tolist()}")
    # 假设最后一列是标签，其余是特征
    columns = data.columns.tolist()
    features_df = data[columns[:-1]]
    labels_df = data[columns[-1]]

# 明确将特征和标签转换为浮点数
# 处理可能存在的非数值数据
features_df = features_df.apply(pd.to_numeric, errors='coerce')
labels_df = pd.to_numeric(labels_df, errors='coerce')

# 检查并处理NaN值
if features_df.isna().any().any() or labels_df.isna().any():
    print("警告：数据中存在NaN值，将使用0填充")
    features_df = features_df.fillna(0)
    labels_df = labels_df.fillna(0)

# 转换为numpy数组并指定数据类型
features = features_df.values.astype(np.float32)
labels = labels_df.values.astype(np.float32)

print(f'features shape: {features.shape}')
print(f'labels shape: {labels.shape}')

# 划分数据集
indices_permutation = np.random.permutation(record_num)
shuffled_features = features[indices_permutation]
shuffled_labels = labels[indices_permutation]
num_validation_samples = int(0.3 * record_num)
# 验证数据集
val_inputs = shuffled_features[:num_validation_samples]
val_labels = shuffled_labels[:num_validation_samples]
# 训练数据集
train_inputs = shuffled_features[num_validation_samples:]
train_labels = shuffled_labels[num_validation_samples:]

print(f'train_inputs shape: {train_inputs.shape}')
print(f'train_labels shape: {train_labels.shape}')
print(f'val_inputs shape: {val_inputs.shape}')
print(f'val_labels shape: {val_labels.shape}')

# 验证数据类型
print(f'train_inputs dtype: {train_inputs.dtype}')
print(f'train_labels dtype: {train_labels.dtype}')
