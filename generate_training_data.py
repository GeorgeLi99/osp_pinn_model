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
    高复杂度函数关系，包含多种非线性特征、多层嵌套和周期性模式，
    专门设计用于训绉50层深度神经网络。
    
    参数:
        X: 形状为 (num_samples, 6) 的numpy数组
        
    返回:
        形状为 (num_samples,) 的numpy数组，值在[0, 1]之间
    """
    # 给特征变量命名
    x1, x2, x3, x4, x5, x6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    
    # 1. 线性项 - 基础线性组合
    linear_term = 0.10 * (0.7*x1 - 1.1*x2 + 1.3*x3 - 0.9*x4 + 1.2*x5 - 0.8*x6)
    
    # 2. 多项式项 - 包含2至5次方
    poly_terms = [
        0.08 * (x1**2 - x2**2 + x3**2 - x4**2 + x5**2 - x6**2),  # 平方项
        0.05 * (x1**3 + x3**3 + x5**3),  # 三次项
        0.02 * (x2**4 - x4**4),  # 四次项
        0.01 * (x1**5 + x6**5)   # 五次项
    ]
    polynomial_term = sum(poly_terms)
    
    # 3. 交互项 - 各种组合的特征交互
    # 3.1 二阶交互
    interaction_2way = 0.12 * (
        x1*x2 + x1*x3 + x1*x4 + x1*x5 + x1*x6 +
        x2*x3 + x2*x4 + x2*x5 + x2*x6 +
        x3*x4 + x3*x5 + x3*x6 +
        x4*x5 + x4*x6 +
        x5*x6
    )
    
    # 3.2 三阶交互
    interaction_3way = 0.06 * (
        x1*x2*x3 + x2*x3*x4 + x3*x4*x5 + x4*x5*x6 +
        x1*x3*x5 + x2*x4*x6
    )
    
    # 3.3 高阶交互
    interaction_high = 0.03 * (x1*x2*x3*x4 + x2*x3*x4*x5 + x3*x4*x5*x6 + x1*x3*x5*x6)
    
    # 组合所有交互项
    interaction_term = interaction_2way + interaction_3way + interaction_high
    
    # 4. 复杂的三角函数组合
    # 4.1 基础三角函数
    trig_basic = 0.12 * (
        np.sin(2*np.pi*x1) + np.cos(3*np.pi*x2) + 
        np.sin(4*np.pi*x3) + np.cos(5*np.pi*x4) +
        np.sin(6*np.pi*x5) + np.cos(7*np.pi*x6)
    )
    
    # 4.2 复合三角函数 - 频率调制
    trig_modulated = 0.10 * (
        np.sin(2*np.pi*x1 * (1 + 0.5*x2)) * np.cos(3*np.pi*x3 * (1 + 0.4*x4)) +
        np.sin(4*np.pi*x2 * (1 + 0.3*x5)) * np.cos(2*np.pi*x4 * (1 + 0.6*x6))
    )
    
    # 4.3 嵌套三角函数 - 高频振荡特征
    trig_nested = 0.08 * (
        np.sin(np.pi * (x1 + np.sin(5*np.pi*x3))) +
        np.cos(np.pi * (x2 + np.cos(6*np.pi*x4))) +
        np.sin(np.pi * (x5 + np.sin(7*np.pi*x6)))
    )
    
    # 5. 复杂的指数和对数函数
    # 5.1 高斯核函数
    gaussian_term = 0.10 * (
        np.exp(-12 * ((x1-0.3)**2 + (x2-0.4)**2)) + 
        np.exp(-10 * ((x3-0.6)**2 + (x4-0.7)**2)) +
        np.exp(-15 * ((x5-0.5)**2 + (x6-0.2)**2))
    )
    
    # 5.2 对数和平方根函数
    log_sqrt_term = 0.07 * (
        np.log(0.1 + x1*x2) + np.sqrt(0.1 + x3*x4) +
        np.log(0.1 + np.sqrt(0.1 + x5*x6))
    )
    
    # 6. 激活函数式配置
    # 6.1 Sigmoid 函数
    sigmoid_term = 0.09 * (
        1 / (1 + np.exp(-10*(x1-0.5))) +
        1 / (1 + np.exp(-12*(x3-0.3))) +
        1 / (1 + np.exp(-15*(x5-0.7)))
    )
    
    # 6.2 Tanh 函数
    tanh_term = 0.08 * (
        np.tanh(4*(x2-0.5)) +
        np.tanh(5*(x4-0.6)) +
        np.tanh(6*(x6-0.3))
    )
    
    # 6.3 ReLU 基础函数
    relu_term = 0.06 * (
        np.maximum(0, 2*(x1-0.3)) +
        np.maximum(0, 3*(x3-0.4)) +
        np.maximum(0, 4*(x5-0.5))
    )
    
    # 7. 复杂的多条件分段函数
    # 7.1 简单分段
    condition1 = (x1 + x2) > 1.0
    condition2 = (x3 + x4) > 1.0
    condition3 = (x5 + x6) > 1.0
    
    # 7.2 生成四种不同的区域响应
    region1 = 0.15 * (x1*x3 + x5) # 区域1的响应
    region2 = 0.15 * (x2*x4 - x6) # 区域2的响应
    region3 = 0.15 * (x1*x6 - x3) # 区域3的响应
    region4 = 0.15 * (x2*x5 + x4) # 区域4的响应
    
    # 7.3 复杂条件逻辑组合
    regional_term = np.zeros_like(x1)
    mask1 = np.logical_and(condition1, condition2)
    mask2 = np.logical_and(~condition1, condition2)
    mask3 = np.logical_and(condition1, ~condition2)
    mask4 = np.logical_and(~condition1, ~condition2)
    
    regional_term = (
        np.where(mask1, region1, 0) +
        np.where(mask2, region2, 0) +
        np.where(mask3, region3, 0) +
        np.where(mask4, region4, 0)
    )
    
    # 8. 深度网络式特征合成 - 模拟多层深度网络的处理
    # 8.1 第一层特征提取
    features_l1 = np.column_stack([
        sigmoid_term, 
        tanh_term, 
        relu_term,
        trig_basic,
        gaussian_term
    ])
    
    # 8.2 第二层特征组合
    features_l2 = np.zeros((X.shape[0], 3))
    features_l2[:, 0] = 0.8 * features_l1[:, 0] + 0.4 * features_l1[:, 1] - 0.3 * features_l1[:, 2]
    features_l2[:, 1] = -0.5 * features_l1[:, 0] + 0.9 * features_l1[:, 3] + 0.2 * features_l1[:, 4]
    features_l2[:, 2] = 0.6 * features_l1[:, 2] - 0.7 * features_l1[:, 3] + 0.5 * features_l1[:, 4]
    
    # 8.3 应用激活函数
    features_l2_act = np.tanh(features_l2)
    
    # 8.4 第三层特征提取
    deep_feature = 0.05 * (
        features_l2_act[:, 0] * trig_modulated +
        features_l2_act[:, 1] * log_sqrt_term +
        features_l2_act[:, 2] * regional_term
    )
    
    # 将所有项组合起来
    y = (
        linear_term + 
        polynomial_term + 
        interaction_term + 
        trig_basic + 
        trig_modulated + 
        trig_nested + 
        gaussian_term + 
        log_sqrt_term + 
        sigmoid_term + 
        tanh_term + 
        relu_term + 
        regional_term + 
        deep_feature
    )
    
    # 高级归一化策略 - 模拟批归一化的效果
    # 首先计算均值和标准差
    batch_mean = np.mean(y)
    batch_std = np.std(y) + 1e-8  # 添加小值防止分母为零
    
    # 标准化
    y_std = (y - batch_mean) / batch_std
    
    # 使用双曼式激活函数 - 结合Tanh和Sigmoid的特性
    # 首先应用tanh压缩到[-1,1]
    y_tanh = np.tanh(y_std)
    
    # 然后转换到[0,1]区间
    y_scaled = (y_tanh + 1) / 2
    
    # 增加异质噪声模式:
    # 1. 高斯噪声 - 基础噪声
    noise_base = np.random.normal(0, 0.005, y_scaled.shape)
    
    # 2. 点态噪声 - 模拟测量设备的不定期干扰
    point_noise_mask = np.random.random(y_scaled.shape) < 0.01  # 1%的概率出现点噪声
    point_noise = np.random.normal(0, 0.03, y_scaled.shape) * point_noise_mask
    
    # 3. 结构性噪声 - 与输入特征相关的噪声
    structured_noise = 0.008 * np.sin(20 * np.pi * (x1 * x2 + x3 * x4))
    
    # 4. 当地噪声 - 与输出值相关的噪声幅度
    local_scale = 0.01 * y_scaled * (1 - y_scaled)  # 在中间值处噪声最大
    local_noise = np.random.normal(0, 1, y_scaled.shape) * local_scale
    
    # 组合所有噪声
    total_noise = noise_base + point_noise + structured_noise + local_noise
    
    # 添加噪声并限制在[0, 1]范围内
    y_with_noise = y_scaled + total_noise
    y_final = np.clip(y_with_noise, 0, 1)
    
    # 复现特定偶然排列模式 - 模拟真实数据中的偶然相关性
    # 适用于测试模型对由于采样错误引起的数据缺陷的适应能力
    for i in range(5):
        idx1 = np.random.randint(0, y_final.shape[0] - 1)
        idx2 = np.random.randint(0, y_final.shape[0] - 1)
        
        if abs(X[idx1, 0] - X[idx2, 0]) < 0.02 and abs(X[idx1, 1] - X[idx2, 1]) < 0.02:
            # 如果两个样本的前两个特征非常相似，则将它们的输出值拉近
            mean_y = (y_final[idx1] + y_final[idx2]) / 2
            y_final[idx1] = mean_y + np.random.normal(0, 0.005)
            y_final[idx2] = mean_y + np.random.normal(0, 0.005)
            # 再次裁剪保证范围
            y_final = np.clip(y_final, 0, 1)
    
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

def generate_and_save_data(num_samples=100000, train_ratio=0.8, output_dir='data'):
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
