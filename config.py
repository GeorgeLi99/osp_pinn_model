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
DATA_PATH = r"C:\0_code\new_osp\data\triple_film_sweep_with_n_results_1.csv"         # 训练数据CSV文件路径
COLUMN = 6                          # 要处理的目标列（索引从0开始）
TEST_DATA_PATH = r"C:\0_code\new_osp\data\triple_film_sweep_with_n_results.csv"  # 测试数据文件路径
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
MODEL_OUTPUT_ACTIVATION = 'tanh'    # 输出层激活函数

# 网络结构配置
MODEL_FIRST_LAYER_UNITS = 24        # 第一隐藏层单元数
MODEL_FIRST_ACTIVATION = 'relu'     # 第一隐藏层激活函数
MODEL_FIRST_DROPOUT = 0.1           # 第一Dropout层的比率

# 第二层 (隐藏层)
MODEL_SECOND_LAYER_UNITS = 32     # 第二层神经元数量
MODEL_SECOND_ACTIVATION = 'relu'  # 第二层激活函数
MODEL_SECOND_DROPOUT = 0.1        # 第二层Dropout比例

# 第三层 (隐藏层)
MODEL_THIRD_LAYER_UNITS = 12      # 第三层神经元数量
MODEL_THIRD_ACTIVATION = 'relu'   # 第三层激活函数
MODEL_THIRD_DROPOUT = 0.1         # 第三层Dropout比例

# 正则化参数
MODEL_KERNEL_INITIALIZER = 'he_normal'  # 权重初始化方法

#------------------------------------------------------------------------------
# 3. 训练参数
#------------------------------------------------------------------------------
# 优化器和训练控制
MODEL_OPTIMIZER = 'adam'            # 优化器: 'adam', 'sgd', 'rmsprop'等
TRAINING_EPOCHS = 3             # 训练轮数
TRAINING_BATCH_SIZE = 256           # 训练批次大小
TRAINING_VALIDATION_SPLIT = 0.3     # 训练集中分出的验证集比例
FIT_VERBOSE = 1                     # 训练过程的输出详细程度: 0=静默, 1=进度条, 2=每轮一行

#------------------------------------------------------------------------------
# 4. 学习率调整参数
#------------------------------------------------------------------------------
# 学习率调整
LR_REDUCE_MONITOR = 'val_loss'      # 监控指标
LR_REDUCE_FACTOR = 0.9              # 学习率降低因子
LR_REDUCE_PATIENCE = 10              # 停止提升的训练轮次
LR_REDUCE_MIN_DELTA = 0.0001        # 最小变化阈值
LR_REDUCE_COOLDOWN = 0              # 冷却期
LR_REDUCE_MIN_LR = 0.0001          # 最小学习率
LR_REDUCE_VERBOSE = 1               # 输出详细程度

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
PINN_PHYSICS_WEIGHT = 0.5          # 物理约束项权重
PINN_SMOOTHNESS_WEIGHT = 0.2       # 平滑约束权重  
PINN_NEGATIVE_PENALTY_WEIGHT = 0.5 # 负值惩罚权重
PINN_DERIVATIVE_WEIGHT = 0.3       # 导数约束权重

# 损失函数选择开关
USE_PINN_LOSS = False              # 是否使用物理信息神经网络损失函数，False则使用Huber损失

# 损失函数配置
PINN_DATA_LOSS_TYPE = 'mse'         # 数据损失类型：'mse'、'mae'

#------------------------------------------------------------------------------
# 7. 硬件加速参数
#------------------------------------------------------------------------------
# GPU配置
GPU_LEARNING_RATE = 5e-3            # GPU训练时的学习率
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