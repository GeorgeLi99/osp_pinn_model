"""
物理信息神经网络 (PINN) 配置文件
包含所有可配置参数，分类整理以便查找和修改
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#------------------------------------------------------------------------------
# 1. 数据处理参数
#------------------------------------------------------------------------------
# 数据路径和列选择
DATA_PATH = r"C:\0_code\new_osp\data\train_data.csv"         # 训练数据CSV文件路径
COLUMN = 6                          # 要处理的目标列（索引从0开始）
TEST_DATA_PATH = r"C:\0_code\new_osp\data\test_data.csv"  # 测试数据文件路径
TEST_DATA_COLUMN = None             # 测试数据目标列，如果为None则使用与训练相同的列

# 数据分割和处理参数
TRAIN_TEST_SPLIT = 0.8              # 训练集与测试集的分割比例
RANDOM_STATE = 42                   # 随机种子，用于可重复的随机操作
NORMALIZE_DATA = True               # 是否对数据进行归一化处理
HANDLE_NAN = 'interpolate'          # NaN处理方式: 'drop', 'zero', 'mean', 'interpolate'

#------------------------------------------------------------------------------
# 2. 模型架构参数
#------------------------------------------------------------------------------
# 输入与输出配置
MODEL_INPUT_SHAPE = (COLUMN,)       # 输入层形状
MODEL_OUTPUT_UNITS = 1              # 输出层单元数
MODEL_OUTPUT_ACTIVATION = 'linear'  # 输出层激活函数 - 改为linear允许任意输出

# 网络结构配置 - 使用更深但更窄的网络架构
# 每层的神经元数量和比例参数
# 第一层 (输入层后的第一个全连接层)
MODEL_LAYER1_UNITS = 56           # units_1: 56
MODEL_LAYER1_ACTIVATION = 'relu'  # 保持激活函数不变
MODEL_LAYER1_DROPOUT = 0.2       # dropout_1: 0.2

# 第二层
MODEL_LAYER2_UNITS = 5            # units_2: 5
MODEL_LAYER2_ACTIVATION = 'relu'  # 激活函数
MODEL_LAYER2_DROPOUT = 0.05        # dropout_2: 0.05

# 第三层
MODEL_LAYER3_UNITS = 18           # units_3: 18
MODEL_LAYER3_ACTIVATION = 'relu'  # 激活函数
MODEL_LAYER3_DROPOUT = 0.0        # dropout_3: 0.0

# 第四层
MODEL_LAYER4_UNITS = 16           # units_4: 16
MODEL_LAYER4_ACTIVATION = 'relu'  # 激活函数
MODEL_LAYER4_DROPOUT = 0.2        # dropout_4: 0.2

# 第五层
MODEL_LAYER5_UNITS = 6            # units_5: 6
MODEL_LAYER5_ACTIVATION = 'relu'  # 激活函数
MODEL_LAYER5_DROPOUT = 0.05       # dropout_5: 0.05

# 第六层
MODEL_LAYER6_UNITS = 4            # units_6: 4
MODEL_LAYER6_ACTIVATION = 'relu'  # 激活函数
MODEL_LAYER6_DROPOUT = 0.2        # dropout_6: 0.2

# DenseNet相关配置已移除

# 正则化参数
MODEL_KERNEL_INITIALIZER = 'random_normal'  # 改为随机正态分布初始化，更适合PINN模型
# 自定义权重初始化参数 - 为random_normal设置更合适的值
MODEL_WEIGHT_INIT_MEAN = 0.1           # 设置为正值，有助于避免输出为零的问题
MODEL_WEIGHT_INIT_STDDEV = 0.05        # 降低标准差防止初始权重过大

#------------------------------------------------------------------------------
# 3. 训练参数
#------------------------------------------------------------------------------
# 优化器和训练控制
MODEL_OPTIMIZER = 'genetic'     # 改用遗传算法优化器，适合复杂的非凸优化问题

# 遗传算法优化器参数 - 基于调优结果
OPTIMIZER_GENETIC_POPULATION_SIZE = 60   # 种群大小 (population_size: 60)
OPTIMIZER_GENETIC_MUTATION_RATE = 0.03   # 变异率 (mutation_rate: 0.03)
OPTIMIZER_GENETIC_CROSSOVER_RATE = 0.85   # 交叉率 (保持不变)
OPTIMIZER_GENETIC_SELECTION_PRESSURE = 1.4 # 选择压力 (保持不变)
OPTIMIZER_GENETIC_LEARNING_RATE = 0.02   # 学习率 (保持不变)

# SGD优化器参数(保留以便切换回去)
OPTIMIZER_SGD_MOMENTUM = 0.9       # SGD优化器动量参数
OPTIMIZER_SGD_DECAY = 1e-5         # SGD优化器衰减率参数

TRAINING_EPOCHS = 100             # 训练轮数
TRAINING_BATCH_SIZE = 128           # 训练批次大小
TRAINING_VALIDATION_SPLIT = 0.3     # 训练集中分出的验证集比例

#------------------------------------------------------------------------------
# 4. 权重监测参数
#------------------------------------------------------------------------------
# 权重可视化开关

# 权重分布直方图及 TensorBoard 记录
ENABLE_WEIGHT_HISTOGRAM = False      # 是否启用权重直方图可视化
WEIGHT_HIST_FREQ = 1                # 每多少个epoch记录一次权重直方图
WEIGHT_HIST_BINS = 50               # 直方图的bins数量

# Matplotlib权重可视化器参数
ENABLE_MATPLOTLIB_VISUALIZER = False  # 是否启用Matplotlib权重可视化器
WEIGHT_VIS_FREQ = 1                 # 每多少个epoch生成一次可视化图

# 神经元权重监测器参数
ENABLE_NEURON_MONITOR = False       # 是否启用神经元权重监测器
NEURON_MONITOR_FREQ = 1             # 每多少个epoch记录一次神经元权重
NEURON_MONITOR_PER_LAYER = 5        # 每层监测的神经元数量
FIT_VERBOSE = 1                     # 训练过程的输出详细程度: 0=静默, 1=进度条, 2=每轮一行

#------------------------------------------------------------------------------
# 4. 学习率调整参数
#------------------------------------------------------------------------------
# 学习率调整
LR_REDUCE_MONITOR = 'val_loss'      # 监控指标
LR_REDUCE_FACTOR = 0.9              # 学习率降低因子
LR_REDUCE_PATIENCE = 5              # 停止提升的训练轮次
LR_REDUCE_MIN_DELTA = 0.0001        # 最小变化阈值
LR_REDUCE_COOLDOWN = 0              # 冷却期
LR_REDUCE_MIN_LR = 0.0001          # 最小学习率
LR_REDUCE_VERBOSE = 1               # 输出详细程度

# 早停参数
EARLY_STOPPING_ENABLED = True       # 是否启用早停
EARLY_STOPPING_MONITOR = 'val_loss' # 监控指标
EARLY_STOPPING_PATIENCE = 10        # 在停止训练前等待的轮数
EARLY_STOPPING_MIN_DELTA = 0.0005   # 最小变化阈值
EARLY_STOPPING_RESTORE_BEST = True  # 是否恢复最佳权重
EARLY_STOPPING_VERBOSE = 1          # 输出详细程度

#------------------------------------------------------------------------------
# 5. K折交叉验证参数
#------------------------------------------------------------------------------
KFOLD_NUM_SPLITS = 5                # 交叉验证折数
KFOLD_SHUFFLE = True                # 是否在分割前打乱数据
KFOLD_RANDOM_SEED = 42              # K折随机种子，确保结果可重现

#------------------------------------------------------------------------------
# 6. 物理信息神经网络（PINN）参数
#------------------------------------------------------------------------------
# PINN损失函数参数
PINN_PHYSICS_WEIGHT = 2.0          # 物理约束项权重
PINN_SMOOTHNESS_WEIGHT = 1.0       # 平滑约束权重 
PINN_NEGATIVE_PENALTY_WEIGHT = 0.5 # 负值惩罚权重（降低以减少输出做能强烈偏向正向）
PINN_DERIVATIVE_WEIGHT = 1.0       # 导数约束权重

# 损失函数选择开关
USE_PINN_LOSS = True              # 是否使用物理信息神经网络损失函数，False则使用Huber损失

# 损失函数配置
PINN_DATA_LOSS_TYPE = 'mse'         # 数据损失类型：'mse'、'mae'

#------------------------------------------------------------------------------
# 7. 硬件加速参数
#------------------------------------------------------------------------------
# GPU配置
GPU_LEARNING_RATE = 0.1            # GPU训练时的学习率
GPU_MEMORY_GROWTH = True            # 是否允许GPU按需分配内存
GPU_MIXED_PRECISION = True          # 是否使用混合精度训练

#------------------------------------------------------------------------------
# 8. 预测和可视化参数
#------------------------------------------------------------------------------
# 文件路径相关
MODELS_BASE_DIR = 'models'          # 模型文件基础目录
MODEL_FILENAME_PREFIX = 'best_pinn_model'  # 模型文件名前缀
MODEL_FORMAT_PRIORITY = ['h5', 'keras']  # 模型格式优先级，按列表顺序尝试

# 获取当前脚本所在目录的绝对路径（更稳健的方式）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 构建所有路径时使用绝对路径并处理斜杠问题
def get_abs_path(rel_path):
    """返回相对于脚本目录的绝对路径，并统一处理斜杠"""
    return os.path.normpath(os.path.join(SCRIPT_DIR, rel_path))

# 默认模型路径（绝对路径）
PRETRAINED_MODEL_PATH = get_abs_path(os.path.join(MODELS_BASE_DIR, f'{MODEL_FILENAME_PREFIX}_2.h5'))

# 自定义模型路径（允许用户指定具体的模型文件路径）
# 如果设置为None，将按照预定义路径顺序尝试查找模型
# 如果设置为具体路径，将直接使用该路径加载模型
CUSTOM_MODEL_PATH = r"C:\0_code\new_osp\models\best_tuned_model.h5"

# 预定义的模型文件备选列表
# .h5格式模型文件（优先）
MODEL_PATHS_H5 = [
    get_abs_path(os.path.join(MODELS_BASE_DIR, f'{MODEL_FILENAME_PREFIX}_2.h5')),
    get_abs_path(os.path.join(MODELS_BASE_DIR, f'{MODEL_FILENAME_PREFIX}.h5'))
]

# .keras格式模型文件（备选）
MODEL_PATHS_KERAS = [
    get_abs_path(os.path.join(MODELS_BASE_DIR, f'{MODEL_FILENAME_PREFIX}_2.keras')),
    get_abs_path(os.path.join(MODELS_BASE_DIR, f'{MODEL_FILENAME_PREFIX}.keras'))
]

# 结果存储路径
PREDICTION_RESULTS_DIR = get_abs_path(os.path.join(MODELS_BASE_DIR, 'prediction_results'))

# 图表和可视化参数
PREDICTION_PLOT_DPI = 300           # 图表DPI
PREDICTION_PLOT_FIGSIZE = (10, 6)   # 预测图表尺寸
PREDICTION_SCATTER_ALPHA = 0.5      # 散点图透明度
PREDICTION_TEXT_POS = (0.05, 0.95)  # 图表中文本位置

# 评估参数
PREDICTION_USE_PHYSICS_INFORMED_EVAL = True  # 是否使用物理约束评估
PREDICTION_PHYSICS_WEIGHT = 0.3     # 预测中使用的物理约束权重
PREDICTION_DIV_ZERO_EPSILON = 1e-10 # 避免除零的小值

#------------------------------------------------------------------------------
# 9. 系统和环境参数
#------------------------------------------------------------------------------
# 文件夹路径
CHECKPOINTS_DIR = get_abs_path('checkpoints')     # 检查点保存目录
LOGS_DIR = get_abs_path('logs')                   # 日志保存目录
MODELS_DIR = get_abs_path(MODELS_BASE_DIR)        # 模型保存目录

# 日志和检查点设置
SAVE_CHECKPOINTS = True             # 是否保存检查点
SAVE_BEST_ONLY = True               # 是否只保存最佳模型
CHECKPOINT_FREQUENCY = 5            # 每多少轮保存一次检查点