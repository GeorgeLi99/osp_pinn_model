"""
训练数据生成器
该脚本用于生成符合要求的训练数据：6个输入变量，1个输出变量。
输出与输入之间存在中等复杂度的函数关系，输出值在0-1范围内。
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置随机种子，确保结果可重现
np.random.seed(42)

def generate_input_features(num_samples, num_features=6):
    """
    生成输入特征数据
    
    参数:
        num_samples: 样本数量
        num_features: 特征数量，默认为6
        
    返回:
        形状为 (num_samples, num_features) 的numpy数组
    """
    # 生成在[0, 1]范围内均匀分布的特征
    features = np.random.uniform(0, 1, size=(num_samples, num_features))
    
    return features

def complex_function(X):
    """
    中等复杂度的函数，将6个输入特征映射到[0, 1]范围内的输出
    
    参数:
        X: 形状为 (num_samples, 6) 的numpy数组
        
    返回:
        形状为 (num_samples,) 的numpy数组，值在[0, 1]之间
    """
    # 解包特征
    x1, x2, x3, x4, x5, x6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    
    # 非线性转换
    f1 = np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2)
    f2 = x3**2 * np.exp(-x4)
    f3 = np.log(1 + x5) / (1 + np.exp(x6 - 0.5))
    f4 = np.tanh(x1 * x3 + x2 * x4)
    f5 = 0.2 * np.sin(3 * np.pi * x5) * np.cos(2 * np.pi * x6)
    
    # 特征交互
    f6 = 0.1 * (x1 * x2 * x3) + 0.05 * (x4 * x5 * x6)
    f7 = 0.15 * np.sqrt(x1 + x3 + x5) * np.sqrt(x2 + x4 + x6)
    
    # 组合所有效应
    y = 0.3 * f1 + 0.15 * f2 + 0.2 * f3 + 0.1 * f4 + 0.1 * f5 + 0.1 * f6 + 0.05 * f7
    
    # 将结果映射到[0, 1]区间
    # 首先计算当前输出的范围
    y_min, y_max = np.min(y), np.max(y)
    
    # 然后将其缩放到[0, 1]区间
    y_scaled = (y - y_min) / (y_max - y_min)
    
    # 添加少量噪声以模拟真实数据
    noise = np.random.normal(0, 0.01, y_scaled.shape)
    y_with_noise = y_scaled + noise
    
    # 确保所有值都在[0, 1]范围内
    y_final = np.clip(y_with_noise, 0, 1)
    
    return y_final

def plot_partial_dependency(X, y, feature_idx, num_points=100, feature_name=None):
    """
    绘制部分依赖图，展示输入特征与输出之间的关系
    
    参数:
        X: 输入特征数组
        y: 输出标签数组
        feature_idx: 要分析的特征索引
        num_points: 用于绘图的点数
        feature_name: 特征名称
    """
    if feature_name is None:
        feature_name = f"特征 {feature_idx + 1}"
    
    # 创建一个新的测试点数组，其中只改变指定特征的值
    X_test = np.zeros((num_points, X.shape[1]))
    
    # 对于所有其他特征，使用训练数据的平均值
    for i in range(X.shape[1]):
        X_test[:, i] = np.mean(X[:, i])
    
    # 对指定特征使用均匀分布的值
    x_range = np.linspace(0, 1, num_points)
    X_test[:, feature_idx] = x_range
    
    # 计算对应的输出值
    y_test = complex_function(X_test)
    
    # 绘制部分依赖图
    plt.figure(figsize=(8, 4))
    plt.plot(x_range, y_test)
    plt.xlabel(feature_name)
    plt.ylabel('输出值')
    plt.title(f'{feature_name}的部分依赖图')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def generate_and_save_data(num_samples, train_ratio=0.8, output_dir='data'):
    """
    生成训练和测试数据，并保存为CSV文件
    
    参数:
        num_samples: 样本总数
        train_ratio: 训练集占总样本的比例
        output_dir: 输出目录
    
    返回:
        train_file_path: 训练数据文件路径
        test_file_path: 测试数据文件路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输入特征
    X = generate_input_features(num_samples)
    
    # 计算输出值
    y = complex_function(X)
    
    # 拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, random_state=42
    )
    
    # 创建训练数据DataFrame
    train_df = pd.DataFrame(
        data=np.column_stack([X_train, y_train[:, np.newaxis]]),
        columns=[f'feature_{i+1}' for i in range(X.shape[1])] + ['target']
    )
    
    # 创建测试数据DataFrame
    test_df = pd.DataFrame(
        data=np.column_stack([X_test, y_test[:, np.newaxis]]),
        columns=[f'feature_{i+1}' for i in range(X.shape[1])] + ['target']
    )
    
    # 保存为CSV文件
    train_file_path = os.path.join(output_dir, 'train_data.csv')
    test_file_path = os.path.join(output_dir, 'test_data.csv')
    
    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)
    
    print(f"生成的训练数据保存至: {train_file_path}")
    print(f"训练样本数: {len(train_df)}")
    print(f"生成的测试数据保存至: {test_file_path}")
    print(f"测试样本数: {len(test_df)}")
    
    # 创建数据可视化
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 为每个特征绘制部分依赖图
    for i in range(X.shape[1]):
        fig = plot_partial_dependency(X, y, i, feature_name=f'特征 {i+1}')
        fig_path = os.path.join(viz_dir, f'feature_{i+1}_dependency.png')
        fig.savefig(fig_path)
        plt.close(fig)
    
    # 绘制目标值分布图
    plt.figure(figsize=(8, 4))
    plt.hist(y, bins=50, alpha=0.7)
    plt.xlabel('目标值')
    plt.ylabel('频率')
    plt.title('目标值分布')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'target_distribution.png'))
    plt.close()
    
    # 创建数据分布散点图（选择前两个特征进行可视化）
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='目标值')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title('特征 1 vs 特征 2 (颜色表示目标值)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'features_scatter.png'))
    plt.close()
    
    print(f"数据可视化保存至: {viz_dir}")
    
    return train_file_path, test_file_path

def analyze_function_properties():
    """
    分析生成函数的特性，打印函数的基本信息
    """
    print("\n函数特性分析:")
    
    # 生成大量样本以分析函数特性
    num_samples = 10000
    X = generate_input_features(num_samples)
    y = complex_function(X)
    
    # 基本统计信息
    print(f"输出均值: {np.mean(y):.4f}")
    print(f"输出标准差: {np.std(y):.4f}")
    print(f"输出中位数: {np.median(y):.4f}")
    print(f"输出最小值: {np.min(y):.4f}")
    print(f"输出最大值: {np.max(y):.4f}")
    
    # 计算每个特征与输出之间的相关性
    correlations = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append(corr)
        print(f"特征 {i+1} 与输出的相关性: {corr:.4f}")
    
    # 找出最具影响力的特征
    abs_correlations = np.abs(correlations)
    most_influential = np.argmax(abs_correlations)
    print(f"最具影响力的特征: 特征 {most_influential + 1} (相关性: {correlations[most_influential]:.4f})")
    
    # 估计函数的复杂度
    # 简单方法：检查多大比例的输出值接近中间值0.5
    middle_range = np.sum((y > 0.45) & (y < 0.55)) / num_samples
    print(f"输出在0.45-0.55范围内的比例: {middle_range:.4f}")

if __name__ == "__main__":
    print("="*80)
    print("训练数据生成器")
    print("="*80)
    
    # 分析函数特性
    analyze_function_properties()
    
    # 生成并保存数据
    num_samples = 50000  # 样本总数
    train_file, test_file = generate_and_save_data(num_samples)
    
    print("\n数据生成完成！")
    print(f"总样本数: {num_samples}")
    print("数据可用于训练物理信息神经网络(PINN)模型。")
