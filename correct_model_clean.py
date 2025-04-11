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
import random
import math
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ====================== 配置参数 ======================
# 路径配置
DEFAULT_DATA_PATH = r'C:\0_code\new_osp\data\train_data.csv'
MODEL_SAVE_DIR = 'models'
RESULTS_SAVE_DIR = 'results'
MODEL_FILENAME = 'simple_linear_model.h5'

# 模型配置
MODEL_PARAMS = {
    'input_shape': (6,),        # 输入特征维度
    'output_units': 1,          # 输出维度
    # 25层神经网络，每层神经元数量逐渐递减
    'hidden_layers': [
        32, 30, 28, 26, 24,     # 前5层
        22, 20, 18, 16, 15,     # 中间5层
        14, 13, 12, 11, 10,     # 中间5层
        9, 8, 7, 6, 5,          # 中间5层
        4, 4, 4, 4, 4           # 后5层
    ],  
    'hidden_activation': 'relu',# 所有隐藏层使用ReLU激活函数
    'output_activation': 'tanh',# 输出层激活函数
    'initializer': 'he_normal', # 权重初始化方法
    'loss': 'mse',              # 损失函数 (均方误差)
    'metrics': ['mae'],         # 评估指标 (平均绝对误差)
    'optimizer': 'adam'         # 优化器
}

# 训练配置
TRAIN_PARAMS = {
    'epochs': 30,               # 增加训练轮数，因为网络更深
    'batch_size': 64,           # 增大批量大小加快训练
    'validation_split': 0.2,    # 验证集比例
    'verbose': 1,               # 显示详细程度
    'initial_learning_rate': 0.01,  # 初始学习率
    'min_learning_rate': 1e-6,      # 最小学习率
    'cooling_rate': 0.95,           # 模拟退火率冷却率
    'temperature_increase_prob': 0.1  # 温度偏移的概率
}

# 可视化配置
VISUALIZATION_PARAMS = {
    'scatter_figsize': (10, 6),  # 散点图大小
    'hist_figsize': (10, 6),     # 直方图大小
    'hist_bins': 30,             # 直方图柱数
}

# 创建必要的目录
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

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
    
    # 获取初始化器
    if params['initializer'] == 'he_normal':
        initializer = tf.keras.initializers.he_normal()
    else:
        initializer = params['initializer']
    
    # 定义输入层
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    
    # 构建隐藏层
    x = inputs
    for i, units in enumerate(hidden_layers):
        x = tf.keras.layers.Dense(
            units,
            activation=hidden_activation,
            kernel_initializer=initializer,
            name=f"hidden_layer_{i+1}"
        )(x)
    
    # 输出层
    outputs = tf.keras.layers.Dense(
        output_units, 
        activation=output_activation,
        kernel_initializer=initializer,
        name="output_layer"
    )(x)
    
    # 创建模型
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Neural_Network_Model")
    
    # 创建优化器对象
    if isinstance(params['optimizer'], str):
        if params['optimizer'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=params.get('initial_learning_rate', 0.01))
        elif params['optimizer'] == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=params.get('initial_learning_rate', 0.01))
        elif params['optimizer'] == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=params.get('initial_learning_rate', 0.01))
        else:
            print(f"Warning: Unknown optimizer string '{params['optimizer']}', falling back to Adam")
            optimizer = tf.keras.optimizers.Adam(learning_rate=params.get('initial_learning_rate', 0.01))
    else:
        # 已经是优化器对象
        optimizer = params['optimizer']
    
    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss=params['loss'],
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

# 模拟退火学习率调度器
class SimulatedAnnealingLearningRateScheduler(Callback):
    """基于模拟退火算法的学习率调度器
    
    模拟退火算法如金属冷却过程：
    1. 初始高温度(高学习率)允许大范围探索
    2. 随着温度降低(学习率减小)，参数更新变得更精细
    3. 偏移机制允许随机升高温度，帮助跳出局部最优解
    """
    def __init__(self, initial_lr=0.01, min_lr=1e-6, cooling_rate=0.95, temp_increase_prob=0.1):
        super(SimulatedAnnealingLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr  # 初始学习率(初始温度)
        self.min_lr = min_lr          # 最小学习率
        self.cooling_rate = cooling_rate  # 冷却率
        self.temp_increase_prob = temp_increase_prob  # 升温概率
        self.current_lr = initial_lr  # 当前学习率
    
    def on_epoch_begin(self, epoch, logs=None):
        # 为第一个周期设置初始学习率
        if epoch == 0:
            # 检查优化器类型并安全地设置学习率
            try:
                if hasattr(self.model.optimizer, 'learning_rate'):
                    tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.initial_lr)
                    print(f"Initial learning rate: {self.initial_lr}")
                    self.current_lr = self.initial_lr
                else:
                    print("Warning: Optimizer does not have a learning_rate attribute. Unable to adjust learning rate.")
            except Exception as e:
                print(f"Warning: Unable to set initial learning rate: {e}")
            return
        
        # 逐渐减小温度(学习率)
        new_lr = self.current_lr * self.cooling_rate
        
        # 偏移机制: 有时增加温度帮助跳出局部最优解
        if random.random() < self.temp_increase_prob:
            # 添加温度偏移，最多增加30%
            temperature_bump = 1 + random.random() * 0.3
            new_lr = new_lr * temperature_bump
            print(f"Epoch {epoch}: Temporarily increasing learning rate by factor of {temperature_bump:.2f}")
        
        # 保证学习率不低于最小值
        new_lr = max(self.min_lr, new_lr)
        
        # 安全地更新学习率
        try:
            if hasattr(self.model.optimizer, 'learning_rate'):
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                self.current_lr = new_lr
                print(f"Epoch {epoch}: Learning rate set to {new_lr:.6f}")
            else:
                print("Warning: Optimizer does not have a learning_rate attribute. Unable to adjust learning rate.")
        except Exception as e:
            print(f"Warning: Unable to update learning rate: {e}")

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
        monitor='val_loss', 
        save_best_only=True, 
        mode='min', 
        verbose=0
    )
    
    # 创建模拟退火学习率调度器
    lr_scheduler = SimulatedAnnealingLearningRateScheduler(
        initial_lr=params.get('initial_learning_rate', 0.01),
        min_lr=params.get('min_learning_rate', 1e-6),
        cooling_rate=params.get('cooling_rate', 0.95),
        temp_increase_prob=params.get('temperature_increase_prob', 0.1)
    )
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_data=validation_data,
        verbose=params['verbose'],
        callbacks=[checkpoint, lr_scheduler]
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
    
    # 绘制训练和验证损失值曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 保存图表
    loss_plot_path = os.path.join(RESULTS_SAVE_DIR, 'loss_history.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"\nLoss history plot saved to: {loss_plot_path}")
    
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

def predict_and_visualize(model, X_test, y_test=None, params=None):
    """使用模型进行预测并可视化结果
    
    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签，默认为None
        params: 可视化参数，默认使用全局配置
        
    Returns:
        预测结果
    """
    # 使用默认配置或传入的参数
    if params is None:
        params = VISUALIZATION_PARAMS
    
    # 进行预测
    print("\nMaking predictions...")
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
        evaluate_predictions(y_test, y_pred, name="Test Set")
        
        # 创建真实值与预测值的散点图
        plt.figure(figsize=params['scatter_figsize'])
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values')
        plt.grid(True)
        scatter_path = os.path.join(RESULTS_SAVE_DIR, 'predictions_scatter.png')
        plt.savefig(scatter_path)
        plt.close()
        print(f"Scatter plot saved to: {scatter_path}")
        
        # 创建值分布直方图
        plt.figure(figsize=params['hist_figsize'])
        plt.hist(y_test, bins=params['hist_bins'], alpha=0.7, label='True Values', color='blue')
        plt.hist(y_pred, bins=params['hist_bins'], alpha=0.5, label='Predicted Values', color='orange')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of True and Predicted Values')
        plt.legend()
        plt.grid(True)
        hist_path = os.path.join(RESULTS_SAVE_DIR, 'error_histogram.png')
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
    
    # Predict and visualize results
    y_pred = predict_and_visualize(model, X_test, y_test)
    
    print("\nModel training and evaluation complete!")
    return model, history

# If run as a script, execute the main function
if __name__ == "__main__":
    main()
