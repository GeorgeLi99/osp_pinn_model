from process_data import train_inputs, train_labels, val_inputs, val_labels
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from config import *
import os
import pandas as pd
import time
import datetime
import io
import matplotlib
import matplotlib.pyplot as plt
from simple_genetic_optimizer import SimpleGeneticOptimizer  # 导入简化版遗传算法优化器
from neuron_monitor import NeuronWeightMonitor  # 导入神经元权重监测器

# 修改字体配置以解决中文乱码
matplotlib.use('Agg')  # 使用非交互式后端
# 尝试使用不同的中文字体配置
try:
    # 设置中文字体，优先使用系统自带字体
    from matplotlib.font_manager import FontProperties
    if os.name == 'nt':  # Windows系统
        font_paths = [
            r'C:\Windows\Fonts\simhei.ttf',  # 黑体
            r'C:\Windows\Fonts\msyh.ttf',    # 微软雅黑
            r'C:\Windows\Fonts\simsun.ttc',  # 宋体
        ]
        # 检查字体文件是否存在并设置
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = FontProperties(fname=font_path)
                matplotlib.rcParams['font.family'] = font.get_name()
                print(f"使用系统字体: {font.get_name()}")
                break
    
    # 如果没有找到系统字体或不是Windows系统，尝试设置通用字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 
                                             'Arial Unicode MS', 'DejaVu Sans', 'NSimSun',
                                             'FangSong', 'STSong', 'STHeiti', 'STKaiti']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    print("已配置matplotlib字体以支持中文显示")
except Exception as e:
    print(f"配置中文字体时出错: {e}")
    print("将使用默认字体，中文可能显示为乱码")

import matplotlib.pyplot as plt
from keras.utils import plot_model
from loss_pinn import *
from sklearn.model_selection import KFold

# DenseNet导入已移除
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# [注意：已移除特征屏蔽层，所有输入特征都直接用于网络的训练]

# 用于调试和验证的层
@keras.utils.register_keras_serializable(package="PINN")
class DebugLayer(tf.keras.layers.Layer):
    """用于调试和验证的层，打印输入数据并原样返回"""
    
    def __init__(self, layer_name="debug", **kwargs):
        super(DebugLayer, self).__init__(**kwargs)
        self.layer_name = layer_name
        
    def call(self, inputs, training=None):
        if training:
            # 仅在第一次训练迭代时打印，避免过多输出
            tf.print(f"\n[{self.layer_name}] 输入形状:", tf.shape(inputs))
            # 打印批次中第一个样本的前6个特征值
            tf.print(f"[{self.layer_name}] 第一个样本特征:", inputs[0, :6])
        return inputs
    
    def get_config(self):
        config = super(DebugLayer, self).get_config()
        config.update({'layer_name': self.layer_name})
        return config

# [注意：已移除特征屏蔽层的验证函数，不再需要验证特征屏蔽]

# 合并所有数据用于K折交叉验证
all_inputs = np.vstack((train_inputs, val_inputs))
all_labels = np.concatenate((train_labels, val_labels))
print(f'全部数据集大小: {all_inputs.shape[0]}个样本')

# GPU配置
print("正在检测并配置GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 允许TensorFlow按需分配显存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, GPU_MEMORY_GROWTH)
            
        # 打印GPU信息
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        print(f"找到GPU: {gpus[0].name}")
        print(f"GPU详情: {gpu_details}")
        
        # 启用混合精度训练 (可显著提高RTX4060的性能)
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"启用混合精度训练: {policy.name}")
        print(f"计算精度: {policy.compute_dtype}")
        print(f"变量存储精度: {policy.variable_dtype}")
        
    except RuntimeError as e:
        print(f"GPU配置错误: {e}")
else:
    print("未检测到GPU，将使用CPU进行训练")

# 记录硬件信息
if gpus:
    try:
        # 尝试解码GPU名称
        device_name = gpus[0].name.decode('utf-8')
    except (UnicodeError, AttributeError):
        try:
            # 如果解码失败，尝试直接转换为字符串
            device_name = str(gpus[0].name)
        except:
            # 如果还是失败，使用安全值
            device_name = "GPU-Unknown"
    device_type = "GPU"
else:
    device_name = "CPU"
    device_type = "CPU"

# 设置TensorFlow日志级别，减少不必要的日志输出
tf.get_logger().setLevel('ERROR')

# 确保models文件夹存在
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# 确保TensorBoard日志文件夹存在
logs_dir = os.path.join(models_dir, 'logs')
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# 定义残差模块类
class ResidualBlock(tf.keras.layers.Layer):
    """残差模块，增强模型对复杂函数的表达能力"""
    
    def __init__(self, units, activation='relu', dropout_rate=0.1, kernel_initializer=None, 
                 name=None, trainable=True, dtype=None, **kwargs):
        # 接受Keras标准参数，包括'trainable'和'dtype'
        super(ResidualBlock, self).__init__(name=name, trainable=trainable, dtype=dtype, **kwargs)
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer
        
        # 定义层
        self.dense1 = None  # 在build中初始化，以便访问输入尺寸
        self.dense2 = None
        self.activation_layer1 = None
        self.activation_layer2 = None
        self.dropout_layer = None
        self.add_layer = None
        self.bn1 = None
        self.bn2 = None
        self.shortcut = None
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # 处理初始化器 - 确保它是一个字符串类型或已实例化的初始化器
        if isinstance(self.kernel_initializer, str):
            # 如果是字符串，直接使用
            initializer = self.kernel_initializer
        elif self.kernel_initializer == 'random_normal' or (hasattr(self.kernel_initializer, '__name__') and self.kernel_initializer.__name__ == 'RandomNormal'):
            # 使用MODEL_WEIGHT_INIT_MEAN和MODEL_WEIGHT_INIT_STDDEV定义的值
            initializer = 'random_normal'  # 简单字符串，可序列化
        else:
            # 默认使用he_normal
            initializer = 'he_normal'
        
        # 初始化各层
        self.dense1 = tf.keras.layers.Dense(
            self.units, 
            kernel_initializer=initializer
        )
        self.dense2 = tf.keras.layers.Dense(
            input_dim,  # 输出尺寸与输入匹配，以便于残差连接
            kernel_initializer=initializer
        )
        self.activation_layer1 = tf.keras.layers.Activation(self.activation)
        self.activation_layer2 = tf.keras.layers.Activation(self.activation)
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.add_layer = tf.keras.layers.Add()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        # 如果输入尺寸与中间层尺寸不同，添加快捷连接映射
        if input_dim != self.units:
            self.shortcut = tf.keras.layers.Dense(
                self.units,
                kernel_initializer=initializer
            )
        
        super(ResidualBlock, self).build(input_shape)
        
    def call(self, inputs, training=False):
        # 保存原始输入用于残差连接
        shortcut = inputs
        
        # 转换分支
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation_layer1(x)
        x = self.dropout_layer(x, training=training)
        
        # 映射回原始尺寸
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        
        # 应用残差连接
        x = self.add_layer([x, shortcut])
        x = self.activation_layer2(x)
        
        return x
        
    def get_config(self):
        """返回层的配置，支持模型序列化"""
        config = super().get_config()
        
        # 处理kernel_initializer以确保可序列化
        if isinstance(self.kernel_initializer, str):
            kernel_init = self.kernel_initializer
        elif hasattr(self.kernel_initializer, '__name__') and self.kernel_initializer.__name__ == 'RandomNormal':
            kernel_init = 'random_normal'
        else:
            # 备选，使用通用的初始化器
            kernel_init = 'he_normal'
            
        config.update({
            "units": self.units,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": kernel_init,
        })
        return config

# K折交叉验证函数
def create_and_train_model():
    """创建和编译带有残差连接的PINN模型"""
    # 创建物理信息神经网络模型 - 改用Functional API
    
    # 确保使用正确的初始化器
    print(f"\n权重初始化参数: mean={MODEL_WEIGHT_INIT_MEAN}, stddev={MODEL_WEIGHT_INIT_STDDEV}")
    
    if MODEL_KERNEL_INITIALIZER == 'random_normal':
        initializer = tf.keras.initializers.RandomNormal(
            mean=MODEL_WEIGHT_INIT_MEAN, 
            stddev=MODEL_WEIGHT_INIT_STDDEV
        )
        print(f"使用RandomNormal初始化器: mean={MODEL_WEIGHT_INIT_MEAN}, stddev={MODEL_WEIGHT_INIT_STDDEV}")
    else:
        initializer = MODEL_KERNEL_INITIALIZER
        print(f"使用默认初始化器: {MODEL_KERNEL_INITIALIZER}")
    
    # 使用Functional API实现模型
    inputs = tf.keras.layers.Input(shape=MODEL_INPUT_SHAPE, name="input_layer")
    
    # 添加调试层（仅在训练时打印）
    x = DebugLayer(layer_name="input_features")(inputs)
    
    # 使用简单的5层全连接网络架构
    print("\n使用简单的5层全连接网络")
    
    # 第一层
    x = layers.Dense(
        MODEL_LAYER1_UNITS, 
        kernel_initializer=initializer,
        use_bias=True,
        name="dense_layer1"
    )(x)
    x = layers.BatchNormalization(name="batch_norm1")(x)
    x = layers.Activation(MODEL_LAYER1_ACTIVATION, name="activation1")(x)
    x = layers.Dropout(MODEL_LAYER1_DROPOUT, name="dropout1")(x)
    
    # 第二层
    x = layers.Dense(
        MODEL_LAYER2_UNITS, 
        kernel_initializer=initializer,
        use_bias=True,
        name="dense_layer2"
    )(x)
    x = layers.BatchNormalization(name="batch_norm2")(x)
    x = layers.Activation(MODEL_LAYER2_ACTIVATION, name="activation2")(x)
    x = layers.Dropout(MODEL_LAYER2_DROPOUT, name="dropout2")(x)
    
    # 第三层
    x = layers.Dense(
        MODEL_LAYER3_UNITS, 
        kernel_initializer=initializer,
        use_bias=True,
        name="dense_layer3"
    )(x)
    x = layers.BatchNormalization(name="batch_norm3")(x)
    x = layers.Activation(MODEL_LAYER3_ACTIVATION, name="activation3")(x)
    x = layers.Dropout(MODEL_LAYER3_DROPOUT, name="dropout3")(x)
    
    # 第四层
    x = layers.Dense(
        MODEL_LAYER4_UNITS, 
        kernel_initializer=initializer,
        use_bias=True,
        name="dense_layer4"
    )(x)
    x = layers.BatchNormalization(name="batch_norm4")(x)
    x = layers.Activation(MODEL_LAYER4_ACTIVATION, name="activation4")(x)
    x = layers.Dropout(MODEL_LAYER4_DROPOUT, name="dropout4")(x)
    
    # 第五层
    x = layers.Dense(
        MODEL_LAYER5_UNITS, 
        kernel_initializer=initializer,
        use_bias=True,
        name="dense_layer5"
    )(x)
    x = layers.BatchNormalization(name="batch_norm5")(x)
    x = layers.Activation(MODEL_LAYER5_ACTIVATION, name="activation5")(x)
    x = layers.Dropout(MODEL_LAYER5_DROPOUT, name="dropout5")(x)
    
    # 输出层
    outputs = layers.Dense(
        MODEL_OUTPUT_UNITS, 
        activation=MODEL_OUTPUT_ACTIVATION, 
        kernel_initializer=initializer,
        name="output_layer"
    )(x)
    
    # 创建模型 - 使用Functional API
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="PINN_Residual_Model")
    
    # 根据配置选择损失函数
    if USE_PINN_LOSS:
        # 使用物理信息损失函数进行编译
        print("使用物理信息神经网络损失函数 (PhysicsInformedLoss)")
        loss_function = PhysicsInformedLoss(physics_weight=PINN_PHYSICS_WEIGHT)
    else:
        # 使用Huber损失函数
        print("使用Huber损失函数")
        loss_function = tf.keras.losses.Huber(delta=1.0)
    
    # 优化器配置 - 根据配置文件选择优化器
    if MODEL_OPTIMIZER.lower() == 'genetic':
        # 使用简化版遗传算法优化器
        print("使用简化版遗传算法优化器 (Simple Genetic Algorithm Optimizer)")
        optimizer = SimpleGeneticOptimizer(
            population_size=OPTIMIZER_GENETIC_POPULATION_SIZE,
            mutation_rate=OPTIMIZER_GENETIC_MUTATION_RATE,
            crossover_rate=OPTIMIZER_GENETIC_CROSSOVER_RATE,
            selection_pressure=OPTIMIZER_GENETIC_SELECTION_PRESSURE,
            learning_rate=OPTIMIZER_GENETIC_LEARNING_RATE if not gpus else GPU_LEARNING_RATE
        )
    elif MODEL_OPTIMIZER.lower() == 'sgd':
        # 使用SGD优化器
        print("使用SGD优化器 (Stochastic Gradient Descent)")
        if gpus:
            # 如果使用GPU，我们可以使用稍微更大的学习率
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=GPU_LEARNING_RATE,
                momentum=OPTIMIZER_SGD_MOMENTUM,
                decay=OPTIMIZER_SGD_DECAY
            )
        else:
            # 非GPU情况下使用默认学习率的SGD
            optimizer = tf.keras.optimizers.SGD(
                momentum=OPTIMIZER_SGD_MOMENTUM,
                decay=OPTIMIZER_SGD_DECAY
            )
    elif MODEL_OPTIMIZER.lower() == 'adam':
        # 使用Adam优化器
        print("使用Adam优化器")
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=GPU_LEARNING_RATE if gpus else 0.001
        )
    else:
        # 默认使用遗传算法优化器
        print(f"未知优化器 '{MODEL_OPTIMIZER}'，默认使用遗传算法优化器")
        optimizer = GeneticOptimizer(
            population_size=OPTIMIZER_GENETIC_POPULATION_SIZE,
            mutation_rate=OPTIMIZER_GENETIC_MUTATION_RATE,
            crossover_rate=OPTIMIZER_GENETIC_CROSSOVER_RATE,
            selection_pressure=OPTIMIZER_GENETIC_SELECTION_PRESSURE,
            learning_rate=OPTIMIZER_GENETIC_LEARNING_RATE if not gpus else GPU_LEARNING_RATE
        )
    
    model.compile(
        loss=loss_function,
        optimizer=optimizer
    )
    
    return model, loss_function, optimizer

# 创建权重监控回调类
class WeightHistogramLogger(tf.keras.callbacks.Callback):
    """在训练过程中记录每个层的权重直方图。"""
    
    def __init__(self, log_dir, freq=1):
        """初始化权重直方图记录器。
        
        参数:
            log_dir: TensorBoard日志目录
            freq: 每隔多少个epoch记录一次权重直方图，默认为1（每个epoch都记录）
        """
        super(WeightHistogramLogger, self).__init__()
        self.log_dir = log_dir
        self.freq = freq
        self.writer = tf.summary.create_file_writer(log_dir)
    
    def on_epoch_end(self, epoch, logs=None):
        """在每个epoch结束时记录权重直方图。"""
        if epoch % self.freq == 0:
            with self.writer.as_default():
                # 记录每个层的权重直方图
                for i, layer in enumerate(self.model.layers):
                    # 跳过没有权重的层
                    if not layer.weights:
                        continue
                        
                    layer_name = layer.name
                    # 记录每个权重矩阵
                    for j, weight in enumerate(layer.weights):
                        weight_name = weight.name.replace(':', '_')
                        # 记录权重直方图
                        tf.summary.histogram(f'layer_{i}/{layer_name}/{weight_name}', weight, step=epoch)
                        
                        # 记录权重的统计信息
                        tf.summary.scalar(f'stats/layer_{i}/{layer_name}/{weight_name}_mean', 
                                        tf.reduce_mean(weight), step=epoch)
                        tf.summary.scalar(f'stats/layer_{i}/{layer_name}/{weight_name}_std', 
                                        tf.math.reduce_std(weight), step=epoch)
                        tf.summary.scalar(f'stats/layer_{i}/{layer_name}/{weight_name}_min', 
                                        tf.reduce_min(weight), step=epoch)
                        tf.summary.scalar(f'stats/layer_{i}/{layer_name}/{weight_name}_max', 
                                        tf.reduce_max(weight), step=epoch)
                self.writer.flush()

# 创建使用matplotlib直接可视化权重的回调类
class MatplotlibWeightVisualizer(tf.keras.callbacks.Callback):
    """使用Matplotlib直接可视化神经网络每层权重的回调
    
    在每个训练周期结束时直接显示每层的权重分布直方图
    """
    def __init__(self, freq=1, bins=50, figsize=(15, 10)):
        """初始化权重可视化器
        
        Args:
            freq: 显示频率，默认为1（每个epoch）
            bins: 直方图的bins数量
            figsize: 图像大小
            wait_time: 显示每个图形的暂停时间（秒）
        """
        super().__init__()
        self.freq = freq
        self.bins = bins
        self.figsize = figsize
        # 记录每层的权重统计信息
        self.weight_stats = {}
        
        # 确保使用可以在非图形界面工作的后端
        # 首先检查当前后端
        current_backend = matplotlib.get_backend()
        print(f"当前Matplotlib后端: {current_backend}")
        
        # 禁用交互式显示，改为保存文件
        self.output_dir = os.path.join(os.getcwd(), 'weight_plots')
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"已初始化Matplotlib权重可视化器 - 将保存图像到目录: {self.output_dir}")
        
    def on_train_begin(self, logs=None):
        """训练开始时调用，创建权重统计记录的字典"""
        # 初始化权重统计记录
        for i, layer in enumerate(self.model.layers):
            if not layer.weights:
                continue
                
            layer_name = f"{i}_{layer.name}"
            self.weight_stats[layer_name] = {}
            
            for j, weight in enumerate(layer.weights):
                weight_name = weight.name.replace(':', '_')
                self.weight_stats[layer_name][weight_name] = {
                    'means': [],
                    'stds': [],
                    'mins': [],
                    'maxs': [],
                    'epochs': []
                }
    
    def on_epoch_end(self, epoch, logs=None):
        """每个训练周期结束时直接实时显示权重直方图"""
        # 每 freq 个epoch显示一次
        if epoch % self.freq != 0:
            return
            
        # 收集各层权重数据
        weight_data = []
        
        # 仅收集主要层的数据，为了简化显示
        for i, layer in enumerate(self.model.layers):
            # 只关注有权重的层，主要是Dense层
            if not layer.weights or 'dense' not in layer.name.lower():
                continue
                
            layer_name = f"{i}_{layer.name}"
            
            # 确保该层在数据字典中存在
            if layer_name not in self.weight_stats:
                self.weight_stats[layer_name] = {}
            
            # 通常每层都有kernel和bias两个权重矩阵
            for j, weight in enumerate(layer.weights):
                weight_name = weight.name.replace(':', '_')
                weight_values = weight.numpy().flatten()  # 展平权重矩阵
                
                # 计算权重统计信息
                mean_val = np.mean(weight_values)
                std_val = np.std(weight_values)
                min_val = np.min(weight_values)
                max_val = np.max(weight_values)
                
                # 确保该权重矩阵在字典中存在
                if weight_name not in self.weight_stats[layer_name]:
                    self.weight_stats[layer_name][weight_name] = {
                        'means': [],
                        'stds': [],
                        'mins': [],
                        'maxs': [],
                        'epochs': []
                    }
                
                # 将统计信息添加到记录中
                weight_stats = self.weight_stats[layer_name][weight_name]
                weight_stats['means'].append(mean_val)
                weight_stats['stds'].append(std_val)
                weight_stats['mins'].append(min_val)
                weight_stats['maxs'].append(max_val)
                weight_stats['epochs'].append(epoch)
                
                # 已记录的数据保存下来，用于显示
                weight_data.append({
                    'layer_index': i,
                    'layer_name': layer.name,
                    'weight_name': weight_name,
                    'weight_values': weight_values,
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'stats': weight_stats
                })
        
        # 没有权重数据则直接返回
        if not weight_data:
            return
            
        # 清除之前的图形
        plt.close('all')
        
        # 第一个图：所有层的权重直方图合集
        fig1 = plt.figure(figsize=self.figsize)
        # 不使用set_window_title，因为这只在交互式后端下有效
        
        # 计算子图布局
        n_layers = len(weight_data)
        n_cols = min(2, n_layers)  # 最多2列
        n_rows = (n_layers + n_cols - 1) // n_cols  # 向上取整
        
        plt.suptitle(f"Epoch {epoch} - 神经网络权重分布", fontsize=16)
        
        # 绘制每一层的权重直方图
        for i, data in enumerate(weight_data):
            ax = plt.subplot(n_rows, n_cols, i+1)
            
            # 绘制直方图
            plt.hist(data['weight_values'], bins=self.bins, alpha=0.7, color='blue')
            plt.axvline(data['mean'], color='red', linestyle='dashed', linewidth=1)
            
            # 添加标题和标签
            plt.title(f"{data['layer_name']} - {data['weight_name'].split('/')[-1]}")
            plt.xlabel("权重值")
            plt.ylabel("频数")
            
            # 添加统计信息
            stats_text = f"均值: {data['mean']:.4f}\n标准差: {data['std']:.4f}\n最小值: {data['min']:.4f}\n最大值: {data['max']:.4f}"
            plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                       va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 留出空间给主标题
        
        # 第二个图：权重随时间变化的趋势图
        # 仅当有足够数据且不是第一个epoch时才显示
        if epoch > 0:
            fig2 = plt.figure(figsize=self.figsize)
            plt.suptitle(f"权重统计随时间变化 (到Epoch {epoch})", fontsize=16)
            
            # 绘制每一层的权重随时间的变化图
            for i, data in enumerate(weight_data):
                stats = data['stats']
                if len(stats['epochs']) <= 1:
                    continue
                    
                ax = plt.subplot(n_rows, n_cols, i+1)
                plt.plot(stats['epochs'], stats['means'], 'r-', label='均值')
                plt.fill_between(stats['epochs'], 
                              np.array(stats['means']) - np.array(stats['stds']),
                              np.array(stats['means']) + np.array(stats['stds']),
                              color='r', alpha=0.2)
                plt.plot(stats['epochs'], stats['mins'], 'g--', label='最小值')
                plt.plot(stats['epochs'], stats['maxs'], 'b--', label='最大值')
                plt.title(f"{data['layer_name']} - {data['weight_name'].split('/')[-1]}")
                plt.xlabel("Epoch")
                plt.ylabel("值")
                plt.legend(loc='best', fontsize='small')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存图表
        fig1_path = os.path.join(self.output_dir, f"epoch_{epoch:03d}_weights_distribution.png")
        plt.figure(fig1.number)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(fig1_path, dpi=150)
        
        # 如果第二个图存在，也保存它
        if epoch > 0:
            fig2_path = os.path.join(self.output_dir, f"epoch_{epoch:03d}_weights_trends.png")
            plt.figure(fig2.number)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(fig2_path, dpi=150)
        
        # 关闭所有图形以释放内存
        plt.close('all')
        
        # 打印简单的状态信息
        if epoch == 0 or epoch % 5 == 0:
            # 安全访问统计数据
            mean_value = "N/A"
            if weight_data and len(weight_data) > 0:
                try:
                    # 取第一个权重的均值
                    mean_value = f"{weight_data[0]['mean']:.4f}"
                except (KeyError, IndexError):
                    pass
            print(f"Epoch {epoch}: 已生成权重分布图并保存到 {self.output_dir}\epoch_{epoch:03d}_*.png - 均值={mean_value}")

# 创建梯度监控回调类
class GradientMonitor(tf.keras.callbacks.Callback):
    """监控训练过程中的梯度大小，帮助诊断梯度消失或爆炸问题。"""
    
    def __init__(self, freq=1):
        """初始化梯度监控器。
        
        参数:
            freq: 每隔多少个epoch记录一次梯度信息，默认为1（每个epoch都记录）
        """
        super(GradientMonitor, self).__init__()
        self.freq = freq
        self.gradient_logs = []
    
    def on_epoch_end(self, epoch, logs=None):
        """在每个epoch结束时记录梯度信息。"""
        # 将epoch转换为整数，并确保每个epoch都能被记录
        epoch_int = int(epoch)
        if epoch_int % self.freq == 0 or epoch == TRAINING_EPOCHS - 1:  # 确保最后一个epoch一定会被记录
            try:
                # 获取模型权重的绝对值统计
                weight_stats = {}
                for layer in self.model.layers:
                    if layer.weights:
                        for weight in layer.weights:
                            weight_np = weight.numpy()
                            layer_name = weight.name.split('/')[0]
                            if 'kernel' in weight.name or 'bias' in weight.name:  # 只分析主要权重
                                if layer_name not in weight_stats:
                                    weight_stats[layer_name] = []
                                
                                weight_abs = np.abs(weight_np)
                                weight_stats[layer_name].append({
                                    'mean': np.mean(weight_abs),
                                    'max': np.max(weight_abs),
                                    'min': np.min(weight_abs),
                                    'std': np.std(weight_abs)
                                })
                
                # 计算整体权重统计
                all_means = [s['mean'] for stats in weight_stats.values() for s in stats]
                all_maxes = [s['max'] for stats in weight_stats.values() for s in stats]
                
                overall_stats = {
                    'epoch': epoch,
                    'mean_weight': np.mean(all_means) if all_means else 0,
                    'max_weight': np.max(all_maxes) if all_maxes else 0,
                    'layer_stats': weight_stats
                }
                
                self.gradient_logs.append(overall_stats)
                
                # 打印权重摘要
                print(f"\nEpoch {epoch} - 权重监控:")
                print(f"  平均权重大小: {overall_stats['mean_weight']:.8f}")
                print(f"  最大权重大小: {overall_stats['max_weight']:.8f}")
                
                # 判断是否存在潜在问题
                if overall_stats['mean_weight'] < 1e-7:
                    print("  警告: 权重过小，可能存在梯度消失问题")
                elif overall_stats['max_weight'] > 1e3:
                    print("  警告: 权重过大，可能存在梯度爆炸问题")
                    
                # 分析权重变化率
                if len(self.gradient_logs) > 1:
                    last_mean = self.gradient_logs[-2]['mean_weight']
                    current_mean = overall_stats['mean_weight']
                    change_rate = abs(current_mean - last_mean) / (last_mean + 1e-10)
                    print(f"  权重变化率: {change_rate:.6f}")
                    
                    if change_rate < 1e-6:
                        print("  警告: 权重变化极小，模型可能已停止学习")
                
            except Exception as e:
                print(f"权重监控出错: {e}")
    
    def get_gradient_summary(self):
        """返回梯度监控的汇总信息。"""
        if not self.gradient_logs:
            return "无权重监控信息记录"
        
        epochs = [log['epoch'] for log in self.gradient_logs]
        means = [log['mean_weight'] for log in self.gradient_logs]
        maxes = [log['max_weight'] for log in self.gradient_logs]
        
        return {
            'epochs': epochs,
            'mean_weights': means,
            'max_weights': maxes
        }

# 执行K折交叉验证
num_folds = KFOLD_NUM_SPLITS
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# 准备存储K折结果的变量
histories = []
val_losses = []
training_times = []
best_model = None
best_val_loss = float('inf')
best_fold_idx = 0

print(f"\n开始{num_folds}折交叉验证...")
fold_idx = 1

# 用于绘图的所有折叠的训练/验证损失
all_train_losses = []
all_val_losses = []

# 临时模型保存路径
temp_best_model_path = os.path.join(models_dir, 'temp_best_model.h5')

# 遍历每个折
for train_idx, val_idx in kfold.split(all_inputs):
    print(f"\n训练第{fold_idx}/{num_folds}折...")
    
    # 获取当前折的训练集和验证集
    fold_train_inputs = all_inputs[train_idx]
    fold_train_labels = all_labels[train_idx]
    fold_val_inputs = all_inputs[val_idx]
    fold_val_labels = all_labels[val_idx]
    
    print(f"训练集大小: {fold_train_inputs.shape[0]}, 验证集大小: {fold_val_inputs.shape[0]}")
    
    # 创建并编译模型
    model, loss_function, optimizer = create_and_train_model()
    
    # 注意：已移除特征屏蔽验证
    
    # 打印模型总结（仅第一折时显示）
    if fold_idx == 1:
        model.summary()
        
        # 生成并保存模型架构图
        try:
            # 创建模型图像目录
            model_viz_dir = os.path.join(models_dir, 'model_visualizations')
            os.makedirs(model_viz_dir, exist_ok=True)
            
            # 生成两种模型结构图：简单版和带层大小的详细版
            model_plot_path = os.path.join(model_viz_dir, 'model_architecture.png')
            plot_model(model, to_file=model_plot_path, show_shapes=False, show_layer_names=True)
            print(f"模型架构图已保存到: {model_plot_path}")
            
            model_plot_with_shapes_path = os.path.join(model_viz_dir, 'model_architecture_with_shapes.png')
            plot_model(model, to_file=model_plot_with_shapes_path, show_shapes=True, show_layer_names=True, 
                       expand_nested=True, dpi=96)
            print(f"详细模型架构图已保存到: {model_plot_with_shapes_path}")
        except Exception as e:
            print(f"生成模型架构图时出错: {e}")
    
    # 创建学习率降低回调函数
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=LR_REDUCE_MONITOR,
        factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE,
        min_delta=LR_REDUCE_MIN_DELTA,
        cooldown=LR_REDUCE_COOLDOWN,
        min_lr=LR_REDUCE_MIN_LR,
        verbose=LR_REDUCE_VERBOSE
    )
    
    # 创建梯度监控回调函数 - 设置为每个epoch都记录
    gradient_monitor = GradientMonitor(freq=1)
    
    # 为当前交叉验证折创建时间戳标记
    log_dir = os.path.join(logs_dir, f"fold_{fold_idx}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    
    # 初始化回调列表
    callbacks_list = []
    
    # 根据配置创建权重直方图记录器回调
    if ENABLE_WEIGHT_HISTOGRAM:
        weight_histogram_logger = WeightHistogramLogger(
            log_dir=log_dir, 
            freq=WEIGHT_HIST_FREQ
        )
        callbacks_list.append(weight_histogram_logger)
        print(f"已启用权重直方图记录器, 频率={WEIGHT_HIST_FREQ} epochs")
    
    # 根据配置创建Matplotlib权重可视化器回调
    if ENABLE_MATPLOTLIB_VISUALIZER:
        matplotlib_visualizer = MatplotlibWeightVisualizer(
            freq=WEIGHT_VIS_FREQ,  # 使用配置的频率
            bins=WEIGHT_HIST_BINS,  # 使用配置的bins数量
            figsize=(15, 10)  # 图像大小固定
        )
        callbacks_list.append(matplotlib_visualizer)
        print(f"已启用Matplotlib权重可视化器, 频率={WEIGHT_VIS_FREQ} epochs")
    
    # 根据配置创建神经元权重监测器
    if ENABLE_NEURON_MONITOR:
        neuron_monitor = NeuronWeightMonitor(
            output_dir='neuron_weights',  # 输出目录
            freq=NEURON_MONITOR_FREQ,  # 使用配置的频率
            max_neurons_per_layer=NEURON_MONITOR_PER_LAYER,  # 每层监测的神经元数量
            layer_name_filter='dense'  # 只监测dense层
        )
        callbacks_list.append(neuron_monitor)
        print(f"已启用神经元权重监测器, 频率={NEURON_MONITOR_FREQ} epochs, 每层{NEURON_MONITOR_PER_LAYER}个神经元")
        
    # 添加早停回调函数
    if EARLY_STOPPING_ENABLED:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=EARLY_STOPPING_MONITOR,
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=EARLY_STOPPING_RESTORE_BEST,
            verbose=EARLY_STOPPING_VERBOSE
        )
        callbacks_list.append(early_stopping)
        print(f"已启用早停功能, 监控={EARLY_STOPPING_MONITOR}, 耐心度={EARLY_STOPPING_PATIENCE} epochs")
    
    # 创建TensorBoard回调
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,  # 每个epoch都记录权重直方图
        write_graph=True,  # 记录模型图结构
        write_images=True,  # 将权重可视化为图像
        update_freq='epoch',  # 每个epoch更新一次日志
        profile_batch=0,  # 禁用分析以提高性能
    )
    
    # 检查和验证初始权重的设置
    print("\n\n=== 检查模型各层初始权重 === ")
    print(f"config.py中的权重初始化设置: MODEL_WEIGHT_INIT_MEAN={MODEL_WEIGHT_INIT_MEAN}, MODEL_WEIGHT_INIT_STDDEV={MODEL_WEIGHT_INIT_STDDEV}")
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'weights') and len(layer.weights) > 0:
            for j, weight in enumerate(layer.weights):
                weight_mean = tf.reduce_mean(weight).numpy()
                weight_std = tf.math.reduce_std(weight).numpy()
                weight_min = tf.reduce_min(weight).numpy()
                weight_max = tf.reduce_max(weight).numpy()
                print(f"\u5c42 {i} ({layer.name}) - 权重 {j} ({weight.name}): \n  均值={weight_mean:.4f}, 标准差={weight_std:.4f}, 最小值={weight_min:.4f}, 最大值={weight_max:.4f}")
    
    # 将模型添加到TensorBoard并记录每个层的权重直方图
    try:
        # 创建TensorBoard的writer
        writer = tf.summary.create_file_writer(log_dir)
        
        # 记录模型概要文本
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary_str = '\n'.join(model_summary)
        
        # 准备用于模型计算图的输入
        sample_input = tf.zeros([1] + list(model.input_shape[1:]))
        
        # 添加模型计算图
        @tf.function
        def traced_model(inputs):
            return model(inputs)
        
        # 使用writer记录及可视化信息
        with writer.as_default():
            # 添加模型概要文本
            tf.summary.text('model_summary', model_summary_str, step=0)
            
            # 为每个层单独记录权重直方图
            for i, layer in enumerate(model.layers):
                # 跳过没有权重的层（如激活函数、dropout等）
                if not layer.weights:
                    continue
                    
                # 记录每个层的名称和每个权重矩阵
                for j, weight in enumerate(layer.weights):
                    weight_name = weight.name.replace(':', '_')
                    # 将权重引用作为标量记录每个权重矩阵的分布
                    tf.summary.histogram(f'layer_{i}/{weight_name}', weight, step=0)
                    
                    # 记录权重的统计信息
                    tf.summary.scalar(f'layer_{i}/{weight_name}_mean', tf.reduce_mean(weight), step=0)
                    tf.summary.scalar(f'layer_{i}/{weight_name}_std', tf.math.reduce_std(weight), step=0)
                    tf.summary.scalar(f'layer_{i}/{weight_name}_min', tf.reduce_min(weight), step=0)
                    tf.summary.scalar(f'layer_{i}/{weight_name}_max', tf.reduce_max(weight), step=0)
            
            # 添加计算图 (这相当于add_graph)
            tf.summary.trace_on(graph=True)
            traced_model(sample_input)
            tf.summary.trace_export(
                name="model_graph", 
                step=0, 
                profiler_outdir=log_dir
            )
            
            # 可选：生成模型架构图也作为图像记录
            buf = io.BytesIO()
            plot_model(model, to_file=buf, show_shapes=True, show_layer_names=True, expand_nested=True)
            buf.seek(0)
            model_image = tf.image.decode_png(buf.getvalue(), channels=4)
            model_image = tf.expand_dims(model_image, 0)
            tf.summary.image('model_architecture', model_image, step=0)
            
        print(f"模型架构图已通过add_graph方式写入TensorBoard日志，可在GRAPHS选项卡中查看")
    except Exception as e:
        print(f"将模型图写入TensorBoard时出错: {e}")
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 训练模型
    # 组合所有回调
    all_callbacks = [reduce_lr, gradient_monitor, tensorboard_callback] + callbacks_list
    
    history = model.fit(
        fold_train_inputs, fold_train_labels, 
        epochs=TRAINING_EPOCHS, 
        batch_size=TRAINING_BATCH_SIZE,
        validation_data=(fold_val_inputs, fold_val_labels),
        callbacks=all_callbacks,  # 使用根据配置生成的回调列表
        verbose=FIT_VERBOSE
    )
    
    # 记录训练结束时间
    end_time = time.time()
    fold_training_time = end_time - start_time
    training_times.append(fold_training_time)
    
    # 存储训练历史以便后续绘图
    histories.append(history)
    
    # 在验证集上评估模型
    val_loss = model.evaluate(fold_val_inputs, fold_val_labels, verbose=0)
    val_losses.append(val_loss)
    
    # 存储训练损失和验证损失用于绘图
    all_train_losses.append(history.history['loss'])
    all_val_losses.append(history.history['val_loss'])
    
    # 如果此折模型是最佳模型，保存它
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_fold_idx = fold_idx
        
        # 保存当前最佳模型到临时文件
        model.save(temp_best_model_path)
        print(f"发现更好的模型（第{fold_idx}折），已保存")
        
        # 保存当前折的梯度信息
        best_gradient_summary = gradient_monitor.get_gradient_summary()
    
    # 显示梯度监控摘要
    gradient_summary = gradient_monitor.get_gradient_summary()
    if isinstance(gradient_summary, dict):
        print("\n当前折梯度监控摘要:")
        last_mean = gradient_summary['mean_weights'][-1] if gradient_summary['mean_weights'] else 0
        last_max = gradient_summary['max_weights'][-1] if gradient_summary['max_weights'] else 0
        print(f"  最终平均权重: {last_mean:.8f}")
        print(f"  最终最大权重: {last_max:.8f}")
    
    print(f"第{fold_idx}折 - 验证损失: {val_loss:.6f}, 训练时间: {fold_training_time:.2f}秒")
    fold_idx += 1

# 加载最佳模型（而非使用deepcopy）
# 添加自定义对象到custom_objects以确保模型加载时能识别
custom_objects = {
    # 添加残差模块类
    'ResidualBlock': ResidualBlock,
    # 已移除FeatureMaskingLayer和DenseNet相关模块
    'SimpleGeneticOptimizer': SimpleGeneticOptimizer,
    # 为了兼容性，作为名称也添加
    'GeneticOptimizer': SimpleGeneticOptimizer,
    # 添加调试层
    'DebugLayer': DebugLayer,
    # 添加物理信息损失函数
    'PhysicsInformedLoss': PhysicsInformedLoss,
    # 添加初始化器
    'RandomNormal': tf.keras.initializers.RandomNormal
}
# 安全加载模型或在失败时重新训练
best_model = None

try:
    if os.path.exists(temp_best_model_path):
        print(f"尝试加载最佳模型（第{best_fold_idx}折）...")
        best_model = tf.keras.models.load_model(temp_best_model_path, compile=False, custom_objects=custom_objects)
        print("模型加载成功!")
except Exception as e:
    print(f"模型加载失败: {e}")
    print("将重新训练新模型，使用残差网络结构")
    best_model = None

# 如果无法加载模型，则创建新模型
if best_model is None:
    # 重新编译模型
    if USE_PINN_LOSS:
        best_loss_function = PhysicsInformedLoss(physics_weight=PINN_PHYSICS_WEIGHT)
    else:
        best_loss_function = tf.keras.losses.Huber(delta=1.0)
    if gpus:
        best_optimizer = tf.keras.optimizers.Adam(learning_rate=GPU_LEARNING_RATE)
    else:
        # 判断优化器类型
        if MODEL_OPTIMIZER.lower() == 'genetic':
            # 使用新的SimpleGeneticOptimizer
            best_optimizer = SimpleGeneticOptimizer(
                population_size=OPTIMIZER_GENETIC_POPULATION_SIZE,
                mutation_rate=OPTIMIZER_GENETIC_MUTATION_RATE,
                crossover_rate=OPTIMIZER_GENETIC_CROSSOVER_RATE,
                selection_pressure=OPTIMIZER_GENETIC_SELECTION_PRESSURE,
                learning_rate=OPTIMIZER_GENETIC_LEARNING_RATE if not gpus else GPU_LEARNING_RATE
            )
        else:
            # 使用其他标准优化器
            best_optimizer = MODEL_OPTIMIZER
    best_model.compile(loss=best_loss_function, optimizer=best_optimizer)
    
    # 将最佳模型保存为最终模型
    best_model.save(os.path.join(models_dir, 'best_pinn_model.h5'))
    
    # 删除临时模型文件
    try:
        os.remove(temp_best_model_path)
    except:
        print("无法删除临时模型文件")
else:
    print("警告：未找到最佳模型文件")

# 使用最佳模型在验证集上生成预测值，并创建可视化比较
print("\n生成验证集预测可视化...")
try:
    # 使用所有原始验证数据进行预测
    val_predictions = best_model.predict(val_inputs, batch_size=TRAINING_BATCH_SIZE, verbose=FIT_VERBOSE)
    
    # 创建真实值vs预测值散点图
    plt.figure(figsize=PREDICTION_PLOT_FIGSIZE)
    plt.scatter(val_labels, val_predictions, alpha=PREDICTION_SCATTER_ALPHA)
    # 添加理想预测线 (y=x)
    plt.plot([np.min(val_labels), np.max(val_labels)], [np.min(val_labels), np.max(val_labels)], 'r--')
    plt.title('验证集: 预测值 vs 真实值')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.grid(True)
    
    # 计算并添加R²和RMSE指标到图表
    r2 = r2_score(val_labels, val_predictions)
    mse = mean_squared_error(val_labels, val_predictions)
    rmse = np.sqrt(mse)
    plt.text(PREDICTION_TEXT_POS[0], PREDICTION_TEXT_POS[1], 
             f"R^2 = {r2:.4f}\nRMSE = {rmse:.4f}", 
             transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'validation_predictions.png'), dpi=PREDICTION_PLOT_DPI)
    plt.close()
    
    # 创建真实值和预测值的分布直方图比较
    plt.figure(figsize=PREDICTION_PLOT_FIGSIZE)
    plt.hist(val_labels.flatten(), bins=30, alpha=0.5, label='真实值')
    plt.hist(val_predictions.flatten(), bins=30, alpha=0.5, label='预测值')
    plt.title('验证集: 真实值与预测值分布对比')
    plt.xlabel('值')
    plt.ylabel('频数')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'validation_distribution.png'), dpi=PREDICTION_PLOT_DPI)
    plt.close()
    
    # 如果有梯度记录，绘制梯度变化图
    if 'best_gradient_summary' in locals() and isinstance(best_gradient_summary, dict):
        plt.figure(figsize=PREDICTION_PLOT_FIGSIZE)
        plt.plot(best_gradient_summary['epochs'], best_gradient_summary['mean_weights'], 'b-', label='平均权重')
        plt.plot(best_gradient_summary['epochs'], best_gradient_summary['max_weights'], 'r-', label='最大权重')
        plt.title('训练权重变化')
        plt.xlabel('Epoch')
        plt.ylabel('权重大小')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, 'weight_monitoring.png'), dpi=PREDICTION_PLOT_DPI)
        plt.close()
        print(f"  - {os.path.join(models_dir, 'weight_monitoring.png')}")
    
    # 保存验证集评估指标
    val_metrics = {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mae': mean_absolute_error(val_labels, val_predictions),
        'rel_error': np.mean(np.abs((val_labels - val_predictions) / (val_labels + PREDICTION_DIV_ZERO_EPSILON))) * 100
    }
    
    # 将评估结果保存到训练日志中
    print(f"\n验证集评估结果:")
    print(f"  均方误差 (MSE): {val_metrics['mse']:.6f}")
    print(f"  均方根误差 (RMSE): {val_metrics['rmse']:.6f}")
    print(f"  平均绝对误差 (MAE): {val_metrics['mae']:.6f}")
    print(f"  R² 分数: {val_metrics['r2']:.6f}")
    print(f"  平均相对误差: {val_metrics['rel_error']:.2f}%")
    
    print(f"\n验证集可视化已保存到:")
    print(f"  - {os.path.join(models_dir, 'validation_predictions.png')}")
    print(f"  - {os.path.join(models_dir, 'validation_distribution.png')}")
    
except Exception as e:
    print(f"生成验证集预测可视化时出错: {e}")

# 计算并显示K折交叉验证结果
mean_val_loss = np.mean(val_losses)
std_val_loss = np.std(val_losses)
total_training_time = np.sum(training_times)

print("\n" + "="*50)
print(f"K折交叉验证结果 (k={num_folds}):")
print(f"平均验证损失: {mean_val_loss:.6f} ± {std_val_loss:.6f}")
print(f"最佳验证损失: {best_val_loss:.6f} (第{best_fold_idx}折)")
print(f"总训练时间: {total_training_time:.2f}秒")

# 打印每折的详细结果
for i, val_loss in enumerate(val_losses):
    print(f"第{i+1}折 - 验证损失: {val_loss:.6f}, 训练时间: {training_times[i]:.2f}秒")

# 格式化总训练时间
hours, remainder = divmod(total_training_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"总训练时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")

# 绘制每折的训练/验证损失曲线
plt.figure(figsize=(15, 10))

for i in range(num_folds):
    # 绘制训练损失
    plt.subplot(2, 1, 1)
    epochs = range(1, len(all_train_losses[i]) + 1)
    plt.plot(epochs, all_train_losses[i], alpha=0.5, linestyle='--', label=f'训练-折{i+1}')
    if i == 0:  # 只在第一折时添加标题和标签，避免重复
        plt.title('训练损失 (所有折)')
        plt.ylabel('损失')
        plt.grid(True)
    
    # 绘制验证损失
    plt.subplot(2, 1, 2)
    plt.plot(epochs, all_val_losses[i], alpha=0.5, linestyle='--', label=f'验证-折{i+1}')
    if i == 0:  # 只在第一折时添加标题和标签，避免重复
        plt.title('验证损失 (所有折)')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.grid(True)

# 绘制平均训练/验证损失
# 由于使用遗传算法优化器，不同折可能有不同的训练轨迹长度
# 判断各折训练历史记录是否长度一致

# 检查所有训练历史的长度
print("\n各折训练历史长度:")
for i, hist in enumerate(all_train_losses):
    print(f"Fold {i+1}: {len(hist)}")
    
# 找到最短的训练历史长度
min_length_train = min(len(hist) for hist in all_train_losses)
min_length_val = min(len(hist) for hist in all_val_losses)
print(f"\n使用最小共同长度进行平均: 训练={min_length_train}, 验证={min_length_val}")

# 截断每个折的历史记录使其长度相同
avg_train_loss = np.mean([np.array(hist[:min_length_train]) for hist in all_train_losses], axis=0)
avg_val_loss = np.mean([np.array(hist[:min_length_val]) for hist in all_val_losses], axis=0)

plt.subplot(2, 1, 1)
epochs = range(1, min_length_train + 1)
plt.plot(epochs, avg_train_loss, 'b-', linewidth=2, label='平均训练损失')
plt.title('平均训练损失')
plt.xlabel('Epoch')
plt.ylabel('损失值')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
val_epochs = range(1, min_length_val + 1)
plt.plot(val_epochs, avg_val_loss, 'r-', linewidth=2, label='平均验证损失')
plt.title('平均验证损失')
plt.xlabel('Epoch')
plt.ylabel('损失值')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(models_dir, 'kfold_loss_curves.png'), dpi=300)
plt.close()  # 关闭图形，释放内存

# 保存K折验证信息到文件
with open(os.path.join(models_dir, 'kfold_training_info.txt'), 'w') as f:
    f.write(f"Model: PINN (Physics-Informed Neural Network) with K-fold Cross Validation\n")
    f.write(f"K value (number of folds): {num_folds}\n")
    f.write(f"Total training time: {int(hours)}hr {int(minutes)}min {seconds:.2f}sec\n")
    f.write(f"Average training time per fold: {np.mean(training_times):.2f}sec\n")
    f.write(f"Total number of epochs: {TRAINING_EPOCHS * num_folds}\n")
    f.write(f"Average validation loss: {mean_val_loss:.6f} ± {std_val_loss:.6f}\n")
    f.write(f"Best validation loss: {best_val_loss:.6f} (fold {best_fold_idx})\n")
    f.write(f"Physical constraint weight: {PINN_PHYSICS_WEIGHT}\n")
    f.write(f"Training batch size: {TRAINING_BATCH_SIZE}\n")
    f.write(f"Optimizer: {type(optimizer).__name__}\n")
    f.write(f"Learning rate: {optimizer.learning_rate.numpy() if hasattr(optimizer, 'learning_rate') else 'default'}\n")
    f.write(f"Hardware: {device_type} - {device_name}\n")
    f.write(f"Mixed precision training: {'yes' if gpus else 'no'}\n")
    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 写入每折的详细结果
    f.write("\nDetailed results for each fold:\n")
    for i, val_loss in enumerate(val_losses):
        f.write(f"Fold {i+1} - Validation loss: {val_loss:.6f}, Training time: {training_times[i]:.2f}sec\n")

# 绘制最佳模型结构并保存
try:
    plot_model(
        best_model, 
        to_file=os.path.join(models_dir, 'best_pinn_model_architecture.png'), 
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=192,
        show_layer_activations=True
    )
    print(f"模型架构图已保存到: {os.path.join(models_dir, 'best_pinn_model_architecture.png')}")
except Exception as e:
    print(f"保存模型架构图时出错: {e}")

print(f"\n最佳模型已保存到: {os.path.join(models_dir, 'best_pinn_model.h5')}")
print(f"K折损失曲线已保存到: {os.path.join(models_dir, 'kfold_loss_curves.png')}")
print(f"K折训练信息已保存到: {os.path.join(models_dir, 'kfold_training_info.txt')}")

# 打印TensorBoard启动指南
print("\n要查看训练详情，请运行以下命令启动TensorBoard:")
print(f"  tensorboard --logdir={logs_dir}")
print("然后在浏览器中访问 http://localhost:6006")

print("\n训练和评估完成!")