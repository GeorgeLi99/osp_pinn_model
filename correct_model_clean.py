# -*- coding: utf-8 -*-
"""
Simple Neural Network Model - 多层神经网络
For testing model capacity and data fitting capability
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.losses import Huber
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 导入模型可视化模块
from model_visualizer import visualize_model_structure

# ====================== 配置参数 ======================
# 路径配置
DEFAULT_DATA_PATH = r'C:\0_code\new_osp\data\triple_film_sweep_with_n_results_1.csv'
TEST_DATA_PATH = r'C:\0_code\new_osp\data\triple_film_sweep_with_n_results.csv'
MODEL_SAVE_DIR = 'models'
RESULTS_SAVE_DIR = 'results'
LOGS_DIR = os.path.join('logs', 'fit')
MODEL_FILENAME = 'simple_linear_model.h5'

# 模型结构配置
MODEL_PARAMS = {
    'input_shape': (6,),        # 输入特征维度
    'output_units': 1,          # 输出维度
    # 50层深度神经网络，采用网络瓶颈结构设计
    'hidden_layers': [
        # 引入阶段 - 逐渐缩小，捕捉各种特征
        128, 112, 98, 86, 76,   # 1-5层，高维度特征提取
        
        # 瓶颈压缩阶段 - 快速缩小到瓶颈
        64, 56, 48, 40, 32,     # 6-10层，特征综合
        28, 24, 20, 18, 16,     # 11-15层，特征压缩
        14, 12, 10, 8,  7,      # 16-20层，进一步压缩
        
        # 瓶颈区 - 维持低维度特征提取
        6, 6, 6, 6, 6,          # 21-25层，核心特征提取
        6, 6, 6, 6, 6,          # 26-30层，高效信息处理
        
        # 扩展阶段 - 逐渐扩大接收域
        7, 8, 10, 12, 14,      # 31-35层，特征扩展
        16, 18, 20, 22, 24,     # 36-40层，扩展信息处理
        
        # 推理阶段 - 收缩到适合输出的大小
        20, 16, 12, 8, 6,       # 41-45层，逐渐聚焦到目标
        5, 4, 3, 2, 2           # 46-50层，精细化输出
    ],
    
    # 采用残差连接和层正规化来增强训练稳定性
    'use_residual': True,      # 启用残差连接，防止梯度消失
    'use_batch_norm': True,    # 使用批正规化，提高训练稳定性
    'residual_frequency': 5,   # 每5层添加一个残差连接
    
    'hidden_activation': 'elu',  # 使用Keras内置的ELU激活函数
    'output_activation': 'linear', # 输出层使用线性激活函数
    'initializer': 'he_normal',  # 权重初始化方法，适合ELU
    'metrics': ['mae', 'mse'],   # 评估指标 (平均绝对误差和均方误差)
    'optimizer': 'adam'          # 优化器
}

# 损失函数配置
LOSS_PARAMS = {
    'function': 'huber',         # 使用Huber损失函数
    'delta': 1.0                # Huber损失函数的delta参数，调整MSE和MAE之间的平衡
}

# 训练配置
TRAIN_PARAMS = {
    'epochs': 100,              # 训练轮数增加以适应更复杂的模型
    'batch_size': 64,           # 增大批量大小以加速训练
    'validation_split': 0.2,    # 验证集比例
    'verbose': 2                # 显示详细程度 (每个epoch一行)
}

# 学习率调度器配置
LR_SCHEDULER_PARAMS = {
    'monitor': 'val_loss',       # 监控验证集上的损失
    'factor': 0.2,              # 学习率降低倍数（降低到原来的0.2倍）
    'patience': 10,              # 10个epoch没有改善就降低学习率
    'min_lr': 1e-6,             # 学习率下限
    'cooldown': 2,              # 冷却期
    'verbose': 1                # 显示详细信息
}

# TensorBoard配置
TENSORBOARD_PARAMS = {
    'histogram_freq': 1,         # 每个周期都计算直方图
    'write_graph': True,        # 写入计算图以便可视化模型结构
    'write_images': True,       # 将权重视为图像写入
    'update_freq': 'epoch',     # 每个周期更新一次
    'profile_batch': 0,         # 禁用分析以避免内存问题
    'embeddings_freq': 1,       # 嵌入可视化频率
    'embeddings_metadata': None # 不指定元数据文件
}

# 模型检查点配置
CHECKPOINT_PARAMS = {
    'monitor': 'val_loss',       # 监控验证集上的损失
    'save_best_only': True,     # 只保存最佳模型
    'mode': 'min',              # 监控指标越小越好
    'verbose': 0                # 不显示详细信息
}

# 可视化配置
VISUALIZATION_PARAMS = {
    'scatter_figsize': (10, 6),  # 散点图大小
    'hist_figsize': (10, 6),    # 直方图大小
    'hist_bins': 30,            # 直方图柱数
    'dpi': 300                  # 图像分辨率
}

# 创建必要的目录
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Create a model with two hidden layers (input -> hidden1 -> hidden2 -> output)
def create_simple_model(input_shape=None, output_units=None, params=None):
    """
    创建一个具有可配置隐藏层的神经网络模型
    
    Args:
        input_shape: 输入特征形状，默认使用全局配置
        output_units: 输出单元数量，默认使用全局配置
        params: 模型参数字典，默认使用全局配置
        
    Returns:
        已编译的模型
    """
    # 使用默认配置或传入的参数
    if params is None:
        params = MODEL_PARAMS
    if input_shape is None:
        input_shape = params['input_shape']
    if output_units is None:
        output_units = params['output_units']
    
    # 获取隐藏层配置
    hidden_layers = params['hidden_layers']
    hidden_activation = params['hidden_activation']
    output_activation = params['output_activation']
    use_residual = params.get('use_residual', False)  # 默认不使用残差连接
    use_batch_norm = params.get('use_batch_norm', False)  # 默认不使用批正规化
    residual_frequency = params.get('residual_frequency', 5)  # 每5层添加一个残差连接
    
    # 获取初始化器
    if params['initializer'] == 'he_normal':
        initializer = tf.keras.initializers.he_normal()
    elif params['initializer'] == 'glorot_normal':
        initializer = tf.keras.initializers.GlorotNormal()
    else:
        initializer = params['initializer']
    
    # 定义输入层
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    
    # 构建隐藏层 - 深度网络带残差连接
    x = inputs
    residual_start = x  # 初始残差连接的起点
    residual_units = None  # 记录残差连接起点的神经元数量
    
    for i, units in enumerate(hidden_layers):
        # 当前层序号，从1开始
        layer_idx = i + 1
        
        # 如果使用批正规化，先应用批正规化
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{layer_idx}")(x)
        
        # 创建当前隐藏层
        x = tf.keras.layers.Dense(
            units,
            activation=hidden_activation,
            kernel_initializer=initializer,
            name=f"hidden_layer_{layer_idx}"
        )(x)
        
        # 处理残差连接
        if use_residual:
            # 记录第一个层的神经元数量，用于残差连接的投影
            if i == 0:
                residual_units = units
                residual_start = x
            
            # 每 residual_frequency 层添加一个残差连接
            if (i > 0) and ((i+1) % residual_frequency == 0):
                # 需要调整维度以匹配当前层
                if residual_units != units:
                    # 创建投影映射将上一个残差层的维度调整为当前层
                    projected_residual = tf.keras.layers.Dense(
                        units, 
                        activation=None,
                        kernel_initializer=initializer,
                        name=f"residual_proj_{layer_idx}"
                    )(residual_start)
                    
                    # 添加残差连接
                    x = tf.keras.layers.add([x, projected_residual], name=f"residual_add_{layer_idx}")
                else:
                    # 维度相同，直接添加
                    x = tf.keras.layers.add([x, residual_start], name=f"residual_add_{layer_idx}")
                
                # 更新当前残差连接起点
                residual_start = x
                residual_units = units
    
    # 如果使用批正规化，在输出层前最后应用一次
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(name="final_batch_norm")(x)
    
    # 输出层 - 使用线性激活函数
    outputs = tf.keras.layers.Dense(
        output_units, 
        activation=output_activation,  # 使用线性激活函数
        kernel_initializer=initializer,
        name="output_layer"
    )(x)
    
    # 创建模型
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Neural_Network_Model")
    
    # 根据配置使用相应的损失函数
    if LOSS_PARAMS['function'] == 'huber':
        # 使用Keras内置的Huber损失函数
        huber_loss = Huber(delta=LOSS_PARAMS['delta'])
        model.compile(
            optimizer=params['optimizer'],
            loss=huber_loss,
            metrics=params['metrics']
        )
    else:
        # 使用其他损失函数
        model.compile(
            optimizer=params['optimizer'],
            loss=LOSS_PARAMS['function'],
            metrics=params['metrics']
        )
    
    return model

# Load data
def load_data(data_path):
    """Load CSV data and split into features and labels"""
    print(f"\nReading data: {data_path}")
    data = pd.read_csv(data_path)
    print(f"Data shape: {data.shape}")
    
    # Assume the last column of the CSV file is the target variable, and the first 6 columns are features
    X = data.iloc[:, :6].values  # First 6 columns as input features
    y = data.iloc[:, -1].values.reshape(-1, 1)  # Last column as target output
    
    return X, y



def train_model(model, X_train, y_train, validation_data=None, params=None):
    """训练模型并返回训练历史
    
    Args:
        model: 要训练的模型
        X_train: 训练特征
        y_train: 训练标签
        validation_data: 验证数据元组 (X_val, y_val)，默认为None
        params: 训练参数，默认使用全局配置
        
    Returns:
        训练历史对象
    """
    # 使用默认配置或传入的参数
    if params is None:
        params = TRAIN_PARAMS
    
    print("\nStarting model training...")
    
    # 设置检查点，保存最佳模型
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, 'checkpoint_best_model.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path, 
        monitor=CHECKPOINT_PARAMS['monitor'], 
        save_best_only=CHECKPOINT_PARAMS['save_best_only'], 
        mode=CHECKPOINT_PARAMS['mode'], 
        verbose=CHECKPOINT_PARAMS['verbose']
    )
    
    # 设置TensorBoard回调 - 增强版本以监控更多信息
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOGS_DIR, current_time)
    
    # 确保日志目录存在且有读写权限
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建用于记录额外指标的文件写入器
    file_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'metrics'))
    file_writer.set_as_default()
    
    # 定义学习率调度回调函数 - 用于记录学习率变化
    class LRTensorBoard(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # 安全获取学习率，兼容不同版本的TensorFlow
            try:
                # 尝试从日志中获取学习率（最安全的方法）
                if logs and 'lr' in logs:
                    lr = logs['lr']
                # 或者尝试直接从优化器获取
                elif hasattr(self.model.optimizer, '_decayed_lr'):
                    # TensorFlow 2.x 某些版本
                    lr = float(self.model.optimizer._decayed_lr(tf.float32).numpy())
                elif hasattr(self.model.optimizer, 'lr'):
                    # 尝试各种可能的访问方式
                    if callable(self.model.optimizer.lr):
                        lr = float(self.model.optimizer.lr().numpy())
                    elif hasattr(self.model.optimizer.lr, 'numpy'):
                        lr = float(self.model.optimizer.lr.numpy())
                    else:
                        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                else:
                    # 如果无法获取，使用默认值
                    lr = 0.001
                    
                # 记录学习率
                tf.summary.scalar('learning_rate', data=lr, step=epoch)
            except Exception as e:
                # 如果获取学习率失败，记录错误但不中断训练
                print(f"无法记录学习率: {e}")
    
    # 使用增强的TensorBoard配置
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=TENSORBOARD_PARAMS['histogram_freq'],
        write_graph=TENSORBOARD_PARAMS['write_graph'],
        write_images=TENSORBOARD_PARAMS['write_images'],
        update_freq=TENSORBOARD_PARAMS['update_freq'],
        profile_batch=TENSORBOARD_PARAMS['profile_batch'],
        embeddings_freq=TENSORBOARD_PARAMS['embeddings_freq'],
        embeddings_metadata=TENSORBOARD_PARAMS['embeddings_metadata']
    )
    
    # 创建模型预测可视化回调 - 用于记录预测分布变化
    class PredictionVisualizer(tf.keras.callbacks.Callback):
        def __init__(self, validation_data, log_dir):
            super().__init__()
            self.validation_data = validation_data
            self.file_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'prediction_dist'))
            
        def on_epoch_end(self, epoch, logs=None):
            x_val, y_val = self.validation_data
            y_pred = self.model.predict(x_val)
            
            with self.file_writer.as_default():
                # 记录预测分布
                tf.summary.histogram('predictions', y_pred, step=epoch)
                tf.summary.histogram('ground_truth', y_val, step=epoch)
                
                # 计算并记录指标
                mse = mean_squared_error(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                tf.summary.scalar('val_r2_score', r2, step=epoch)
                tf.summary.scalar('val_mse_detailed', mse, step=epoch)
                tf.summary.scalar('val_mae_detailed', mae, step=epoch)
    
    print(f"TensorBoard 日志目录: {log_dir}")
    print("启动TensorBoard: tensorboard --logdir=logs/fit")
    print("可视化更多指标: 在TensorBoard界面中切换到'SCALARS'、'HISTOGRAMS'、'GRAPHS'等选项卡")
    
    # 创建学习率调度器 - 当验证指标停止改善时自动降低学习率
    reduce_lr = ReduceLROnPlateau(
        monitor=LR_SCHEDULER_PARAMS['monitor'],
        factor=LR_SCHEDULER_PARAMS['factor'],
        patience=LR_SCHEDULER_PARAMS['patience'],
        min_lr=LR_SCHEDULER_PARAMS['min_lr'],
        cooldown=LR_SCHEDULER_PARAMS['cooldown'],
        verbose=LR_SCHEDULER_PARAMS['verbose']
    )
    
    # 初始化回调列表
    callbacks = [
        checkpoint, 
        tensorboard_callback,
        LRTensorBoard(),
        reduce_lr               # 添加学习率调度器
    ]
    
    # 如果有验证数据，添加预测可视化回调
    if validation_data is not None:
        prediction_visualizer = PredictionVisualizer(validation_data, log_dir)
        callbacks.append(prediction_visualizer)
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_data=validation_data,
        shuffle=True,
        verbose=params['verbose'],
        callbacks=callbacks
    )
    
    # 加载最佳模型
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print("Loaded weights from best model checkpoint")
    
    # 保存最终模型
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # 可视化训练历史
    plot_training_history(history)
    
    return history

# Evaluate model prediction results
def plot_training_history(history):
    """可视化训练过程中的损失值和评估指标变化
    
    Args:
        history: 模型训练返回的历史对象
    """
    # 创建结果目录（如果不存在）
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    
    # 绘制训练和验证损失值曲线 - 分两个图，一个显示全范围，一个显示缩放的范围
    
    # 准备绘图数据
    train_loss = history.history['loss']
    val_loss = history.history.get('val_loss', None)
    epochs = range(1, len(train_loss) + 1)
    
    # --------- 第一幅图: 对近期的损失值进行缩放处理 ---------
    plt.figure(figsize=(12, 10))
    
    # 创建两个子图，上面一个显示全部范围
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    if val_loss:
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Model Loss - Full Range')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    # 下面的子图显示缩放后的范围，聚焦于小值
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    if val_loss:
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
        
    # 计算后半段损失值的范围，更聚焦于训练后期
    # 使用训练后半段数据来计算缩放范围
    half_point = len(train_loss) // 2
    recent_train_loss = train_loss[half_point:]
    
    if val_loss:
        recent_val_loss = val_loss[half_point:]
        # 结合两个损失数据计算最适y轴范围
        all_recent_losses = recent_train_loss + recent_val_loss
    else:
        all_recent_losses = recent_train_loss
    
    if all_recent_losses:
        # 计算合适的y轴下限（缩小范围）
        min_loss = min(all_recent_losses)
        # 计算上限：使用最近95%百分位的值或者下回到后半段的最大值
        p95 = sorted(all_recent_losses)[int(len(all_recent_losses) * 0.95)]
        max_loss = min(max(all_recent_losses), p95 * 1.5)
        
        # 稍微扩大一点范围以便观察
        y_min = max(0, min_loss * 0.8)  # 确保下限不低于0
        y_max = max_loss * 1.2
        
        # 设置y轴范围
        plt.ylim(y_min, y_max)
    
    plt.title('Model Loss - Zoomed View (Focus on Later Epochs)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 保存详细视图图表
    loss_plot_path = os.path.join(RESULTS_SAVE_DIR, 'loss_history.png')
    plt.savefig(loss_plot_path, dpi=VISUALIZATION_PARAMS['dpi'])
    plt.close()
    
    # --------- 第二幅图: 增加对数尺度图 ---------
    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, train_loss, 'b-', label='Training Loss')
    if val_loss:
        plt.semilogy(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Model Loss During Training (Log Scale)')
    plt.ylabel('Loss (log scale)')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="-")
    
    # 保存对数尺度图表
    log_loss_plot_path = os.path.join(RESULTS_SAVE_DIR, 'loss_history_log_scale.png')
    plt.savefig(log_loss_plot_path, dpi=VISUALIZATION_PARAMS['dpi'])
    plt.close()
    
    print(f"\nLoss history plots saved to: \n1. {loss_plot_path} \n2. {log_loss_plot_path}")
    
    # 如果有其他指标，也绘制出来
    # 检查是否有评估指标如MAE
    metrics = [m for m in history.history.keys() if not m.startswith('val_') and m != 'loss']
    if metrics:
        plt.figure(figsize=(10, 6))
        for metric in metrics:
            plt.plot(history.history[metric], label=f'Training {metric.upper()}')
            if f'val_{metric}' in history.history:
                plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.upper()}')
        plt.title('Model Metrics During Training')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True)
        
        # 保存图表
        metrics_plot_path = os.path.join(RESULTS_SAVE_DIR, 'metrics_history.png')
        plt.savefig(metrics_plot_path)
        plt.close()
        print(f"Metrics history plot saved to: {metrics_plot_path}")

def evaluate_predictions(y_true, y_pred, name=""):
    """Calculate and print prediction evaluation metrics"""
    # Calculate various evaluation metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Print evaluation metrics
    print(f"\n{name} Evaluation Results:")
    print(f"  Mean Squared Error (MSE): {mse:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  Coefficient of Determination (R²): {r2:.6f}")
    
    return mse, rmse, mae, r2

def predict_and_visualize(model, X_test, y_test=None, params=None, set_name="Test"):
    """使用模型进行预测并可视化结果
    
    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签，默认为None
        params: 可视化参数，默认使用全局配置
        set_name: 数据集名称，用于文件名和图标题，默认为"Test"
        
    Returns:
        预测结果
    """
    # 使用默认配置或传入的参数
    if params is None:
        params = VISUALIZATION_PARAMS
    
    # 进行预测
    print(f"\nMaking predictions for {set_name} Set...")
    start_time = pd.Timestamp.now()
    y_pred = model.predict(X_test)
    end_time = pd.Timestamp.now()
    prediction_time = (end_time - start_time).total_seconds()
    print(f"Prediction time: {prediction_time:.4f} seconds")
    print(f"Prediction shape: {y_pred.shape}")
    
    # 如果有测试标签，计算指标并可视化
    if y_test is not None:
        # 直接使用原始预测值进行评估，不做tanh输出适配
        # 评估预测性能
        evaluate_predictions(y_test, y_pred, name=f"{set_name} Set")
        
        # 创建真实值与预测值的散点图
        plt.figure(figsize=params['scatter_figsize'])
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'True vs Predicted Values ({set_name} Set)')
        plt.grid(True)
        scatter_path = os.path.join(RESULTS_SAVE_DIR, f'predictions_scatter_{set_name.lower()}.png')
        plt.savefig(scatter_path)
        plt.close()
        print(f"Scatter plot saved to: {scatter_path}")
        
        # 创建值分布直方图
        plt.figure(figsize=params['hist_figsize'])
        plt.hist(y_test, bins=params['hist_bins'], alpha=0.7, label='True Values', color='blue')
        plt.hist(y_pred, bins=params['hist_bins'], alpha=0.5, label='Predicted Values', color='orange')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of True and Predicted Values ({set_name} Set)')
        plt.legend()
        plt.grid(True)
        hist_path = os.path.join(RESULTS_SAVE_DIR, f'error_histogram_{set_name.lower()}.png')
        plt.savefig(hist_path)
        plt.close()
        print(f"Distribution histogram saved to: {hist_path}")
    
    return y_pred
    
    return y_pred

def main(data_path=None):
    """主函数 - 创建、训练和评估模型"""
    # 如果未指定数据路径，使用默认路径
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    
    # 打印模型信息
    print("="*80)
    hidden_layers_str = ' -> '.join([f"Hidden[{units}]" for units in MODEL_PARAMS['hidden_layers']])
    print(f"Neural Network Model (Input -> {hidden_layers_str} -> Output[{MODEL_PARAMS['output_activation']}])")
    print("="*80)
    
    # Load data
    X, y = load_data(data_path)
    
    # Print data statistics
    print("\nData Statistics:")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    
    # 创建模型
    model = create_simple_model(input_shape=(X.shape[1],))
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # 可视化模型结构
    visualize_model_structure(model, save_path=os.path.join(RESULTS_SAVE_DIR, "model_structure.png"))
    
    # Print model output layer information
    print("\nModel Output Layer Information:")
    try:
        output_layer = model.layers[-1]
        print(f"Output layer name: {output_layer.name}")
        print(f"Output layer activation: {output_layer.activation.__name__ if hasattr(output_layer.activation, '__name__') else str(output_layer.activation)}")
        print(f"Output layer config: {output_layer.get_config()}")
    except Exception as e:
        print(f"Error getting model output layer information: {e}")
    
    # Train model (assuming 80% of data for training, 20% for testing)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 训练模型
    history = train_model(
        model, 
        X_train, y_train,
        validation_data=(X_test, y_test)
        # 使用全局TRAIN_PARAMS配置，不再直接传入epochs和batch_size
        # 如需自定义参数，可添加: params={‘epochs’: 50, ‘batch_size’: 32}
    )
    
    # Plot training & validation loss values
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    loss_plot_path = os.path.join('results', 'loss_history.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"\nLoss history plot saved to: {loss_plot_path}")
    
    # 预测并可视化训练集结果
    print("\n" + "-"*40)
    print("Evaluating on Training Set:")
    predict_and_visualize(model, X_train, y_train, set_name="Train")
    
    # 预测并可视化验证集结果
    print("\n" + "-"*40)
    print("Evaluating on Validation Set:")
    predict_and_visualize(model, X_test, y_test, set_name="Validation")
    
    # 预测并可视化独立测试集结果
    print("\n" + "-"*40)
    print("Evaluating on External Test Set:")
    test_data = pd.read_csv(TEST_DATA_PATH)
    X_test_data = test_data.iloc[:, :-1].values
    y_test_data = test_data.iloc[:, -1].values
    predict_and_visualize(model, X_test_data, y_test_data, set_name="Test")
    
    print("\nModel training and evaluation complete!")
    return model, history

# If run as a script, execute the main function
if __name__ == "__main__":
    main()
