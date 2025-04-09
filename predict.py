"""
Physics-Informed Neural Network (PINN) Model Prediction Script
This script loads a trained PINN model and makes predictions on specified test data.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import *
from loss_pinn import PhysicsInformedLoss

# 设置matplotlib中文字体支持
import matplotlib
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

print("="*80)
print("PINN Model Prediction")
print("="*80)

# 打印当前工作目录信息（用于调试）
print(f"Current working directory: {os.getcwd()}")

# 创建结果目录
os.makedirs(PREDICTION_RESULTS_DIR, exist_ok=True)

# 检查测试数据文件是否存在
if not os.path.exists(TEST_DATA_PATH):
    print(f"Warning: Test data file not found at {TEST_DATA_PATH}")
    print("Please place your test data file at this location or update TEST_DATA_PATH in config.py")
    exit(1)

# 获取模型文件路径 - 专注于.h5格式
print("只搜索.h5格式模型文件...")

# 如果设置了自定义模型路径，则优先使用它
if CUSTOM_MODEL_PATH is not None:
    print(f"正在使用自定义模型路径: {CUSTOM_MODEL_PATH}")
    found_model_path = CUSTOM_MODEL_PATH if os.path.isfile(CUSTOM_MODEL_PATH) else None
    if not found_model_path:
        print(f"警告: 自定义模型路径 {CUSTOM_MODEL_PATH} 不存在或不是文件")
        print("将尝试使用预定义的模型路径...") 

# 如果没有自定义模型路径或自定义路径无效，则使用预定义路径
if CUSTOM_MODEL_PATH is None or (CUSTOM_MODEL_PATH is not None and not os.path.isfile(CUSTOM_MODEL_PATH)):
    # 定义.h5模型路径列表
    model_paths = []
    # 预定义的.h5模型路径
    h5_model_paths = [
        os.path.abspath(os.path.join(os.getcwd(), MODELS_BASE_DIR, f'{MODEL_FILENAME_PREFIX}_2.h5')),
        os.path.abspath(os.path.join(os.getcwd(), MODELS_BASE_DIR, f'{MODEL_FILENAME_PREFIX}.h5')),
        os.path.join(MODELS_BASE_DIR, f'{MODEL_FILENAME_PREFIX}_2.h5'),
        os.path.join(MODELS_BASE_DIR, f'{MODEL_FILENAME_PREFIX}.h5'),
        'models/best_pinn_model_2.h5',
        'models/best_pinn_model.h5'
    ]
    model_paths.extend(h5_model_paths)

    # 确保PRETRAINED_MODEL_PATH也在搜索列表中（如果是.h5格式）
    if PRETRAINED_MODEL_PATH.lower().endswith('.h5') and PRETRAINED_MODEL_PATH not in model_paths:
        model_paths.insert(0, PRETRAINED_MODEL_PATH)  # 放在最前面

# 如果使用预定义路径（没有有效的自定义路径），则标准化路径并查找模型
if CUSTOM_MODEL_PATH is None or (CUSTOM_MODEL_PATH is not None and not os.path.isfile(CUSTOM_MODEL_PATH)):
    # 标准化所有路径（Windows系统中解决斜杠方向问题）
    model_paths = [os.path.normpath(path) for path in model_paths]
    # 移除重复路径
    model_paths = list(dict.fromkeys(model_paths))

    # 尝试查找可用的模型文件
    found_model_path = None
    for path in model_paths:
        print(f"检查模型路径: {path}")
        if os.path.isfile(path):
            found_model_path = path
            print(f"找到模型文件: {found_model_path}")
            break

if not found_model_path:
    # 如果没有找到预定义路径中的文件，尝试在MODELS_BASE_DIR目录中查找.h5文件
    print(f"在指定路径中找不到模型文件，尝试在{MODELS_BASE_DIR}目录中查找.h5文件...")
    
    found_models = glob.glob(f'{MODELS_BASE_DIR}/*.h5')
    
    if found_models:
        found_model_path = found_models[0]
        print(f"找到替代模型文件: {found_model_path}")
    else:
        print(f"错误: 在{MODELS_BASE_DIR}目录中找不到任何.h5模型文件")
        if os.path.exists(MODELS_BASE_DIR):
            print(f"{MODELS_BASE_DIR}目录内容: {os.listdir(MODELS_BASE_DIR)}")
        exit(1)

# 获取最终的模型路径
model_path = found_model_path
print(f"将使用模型: {model_path}")
start_time = time.time()

# 定义自定义对象字典
custom_objects = {
    "PhysicsInformedLoss": PhysicsInformedLoss
}

# 根据Keras文档优化加载.h5模型的方式
print("加载.h5格式模型文件...")
try:
    # 使用keras.models.load_model并传递custom_objects
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False  # 先不编译，后面手动编译
    )
    
    # 根据配置选择损失函数
    if USE_PINN_LOSS:
        print("使用物理信息神经网络损失函数 (PhysicsInformedLoss)")
        loss_function = PhysicsInformedLoss(
            physics_weight=PINN_PHYSICS_WEIGHT
        )
    else:
        print("使用Huber损失函数")
        loss_function = tf.keras.losses.Huber(delta=1.0)
    
    # 手动编译模型，应用选择的损失函数
    model.compile(
        optimizer='adam',
        loss=loss_function
    )
    print("模型加载并编译成功!")
except Exception as e:
    print(f"模型加载失败: {e}")
    print("请确保模型文件存在且格式正确")
    exit(1)

loading_time = time.time() - start_time
print(f"模型加载时间: {loading_time:.2f} 秒")

# 打印模型摘要
print("\n模型摘要:")
model.summary()

# 打印模型输出层信息
print("\n模型输出层信息:")
try:
    output_layer = model.layers[-1]
    print(f"输出层名称: {output_layer.name}")
    print(f"输出层激活函数: {output_layer.activation.__name__ if hasattr(output_layer.activation, '__name__') else str(output_layer.activation)}")
    print(f"输出层配置: {output_layer.get_config()}")
except Exception as e:
    print(f"获取模型输出层信息时出错: {e}")

# 从测试数据CSV文件读取数据
print(f"\n读取测试数据: {TEST_DATA_PATH}")
test_data = pd.read_csv(TEST_DATA_PATH)
print(f"测试数据形状: {test_data.shape}")

# 确定要使用的列
# 检查COLUMN和TEST_DATA_COLUMN是否为索引数字或列名
is_column_index = isinstance(COLUMN, int)
is_test_column_index = isinstance(TEST_DATA_COLUMN, int) if TEST_DATA_COLUMN is not None else False

# 根据配置确定目标列
if TEST_DATA_COLUMN is not None:
    target_column = TEST_DATA_COLUMN
else:
    target_column = COLUMN

print(f"目标列配置值: {target_column}")

# 获取列名或索引
if is_column_index or is_test_column_index:
    # 如果目标列是索引，打印出索引和对应的列名
    print("目标列是索引值，确定对应的列名...")
    if target_column < len(test_data.columns):
        column_name = test_data.columns[target_column]
        print(f"索引 {target_column} 对应的列名是: {column_name}")
        target_column = column_name
    else:
        print(f"错误: 索引 {target_column} 超出了数据列数 {len(test_data.columns)}")
        print(f"可用列: {', '.join(test_data.columns)}")
        exit(1)

print(f"使用目标列: {target_column}")

# 打乱测试数据
print("\n打乱测试数据...")
# 使用固定的随机种子以确保结果可重现
np.random.seed(RANDOM_STATE)
shuffled_indices = np.random.permutation(len(test_data))
test_data = test_data.iloc[shuffled_indices].reset_index(drop=True)
print(f"数据已随机打乱，使用随机种子: {RANDOM_STATE}")

# 检查目标列是否存在
if target_column not in test_data.columns:
    print(f"错误: 目标列 '{target_column}' 不在测试数据中")
    print(f"可用列: {', '.join(test_data.columns)}")
    exit(1)

# 提取输入特征
test_features = test_data.copy()

# 如果测试数据包含标签列，则提取用于评估
has_labels = True
try:
    test_labels = test_features[target_column].values.reshape(-1, 1)
    # 移除标签列以准备输入特征
    test_features = test_features.drop(columns=[target_column])
except Exception as e:
    print(f"注意: 提取目标列 '{target_column}' 时出错: {e}")
    print("将进行预测而不进行评估")
    has_labels = False

# 预处理输入特征（确保数据格式匹配模型期望）
test_inputs = test_features.values
print(f"测试输入形状: {test_inputs.shape}")

# 添加输入数据的统计信息
print("\n输入特征分析:")
print(f"特征数量: {test_inputs.shape[1]}")
print(f"样本数量: {test_inputs.shape[0]}")

# 打印每个特征的统计信息
print("\n输入特征统计信息:")
for i in range(test_inputs.shape[1]):
    feature_name = test_features.columns[i] if i < len(test_features.columns) else f"特征{i}"
    feature_values = test_inputs[:, i]
    print(f"\n特征 {i}: {feature_name}")
    print(f"  最小值: {np.min(feature_values):.6f}")
    print(f"  最大值: {np.max(feature_values):.6f}")
    print(f"  均值: {np.mean(feature_values):.6f}")
    print(f"  标准差: {np.std(feature_values):.6f}")
    print(f"  前5个值: {feature_values[:5]}")

# 打印前几个完整样本
print("\n前3个完整输入样本:")
for i in range(min(3, test_inputs.shape[0])):
    print(f"样本 {i}:")
    for j in range(test_inputs.shape[1]):
        feature_name = test_features.columns[j] if j < len(test_features.columns) else f"特征{j}"
        print(f"  {feature_name}: {test_inputs[i, j]:.6f}")

# 定义评估指标函数
def evaluate_predictions(y_true, y_pred, name="Dataset"):
    """计算并打印各种评估指标"""
    # 根据配置决定使用标准评估还是物理信息评估
    if PREDICTION_USE_PHYSICS_INFORMED_EVAL and USE_PINN_LOSS:
        # 使用物理信息损失函数计算损失
        pinn_loss = PhysicsInformedLoss(physics_weight=PREDICTION_PHYSICS_WEIGHT)
        loss = pinn_loss(y_true, y_pred).numpy()
        mse = loss
        rmse = np.sqrt(loss)
        mae = np.mean(np.abs(y_true - y_pred))
    else:
        # 计算标准指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
    
    # 计算R²
    r2 = r2_score(y_true, y_pred)
    
    # 计算平均相对误差
    rel_errors = np.abs((y_true - y_pred) / (y_true + PREDICTION_DIV_ZERO_EPSILON))
    mean_rel_error = np.mean(rel_errors) * 100  # 转换为百分比
    
    # 打印结果
    print(f"\n{name} 评估指标:")
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"均方根误差 (RMSE): {rmse:.6f}")
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    print(f"R² 分数: {r2:.6f}")
    print(f"平均相对误差: {mean_rel_error:.2f}%")
    
    # 返回评估指标字典
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'rel_error': mean_rel_error
    }

# 进行预测
print("\n进行预测...")
test_predictions = model.predict(test_inputs)
print(f"预测形状: {test_predictions.shape}")

# 添加预测值分析
print("\n预测值分布分析:")
print(f"最小值: {np.min(test_predictions):.6f}")
print(f"最大值: {np.max(test_predictions):.6f}")
print(f"均值: {np.mean(test_predictions):.6f}")
print(f"中位数: {np.median(test_predictions):.6f}")
print(f"标准差: {np.std(test_predictions):.6f}")
print(f"值域范围: {np.max(test_predictions) - np.min(test_predictions):.6f}")
print(f"前10个预测值: {test_predictions[:10].flatten()}")

# 打印模型最后一层激活函数信息
print("\n模型输出层信息:")
try:
    output_layer = model.layers[-1]
    print(f"输出层名称: {output_layer.name}")
    print(f"输出层激活函数: {output_layer.activation.__name__ if hasattr(output_layer.activation, '__name__') else output_layer.activation}")
    print(f"输出层配置: {output_layer.get_config()}")
except Exception as e:
    print(f"获取模型输出层信息时出错: {e}")

# 如果有标签，进行评估并比较真实值与预测值的分布
if has_labels:
    print("\n真实值分布分析:")
    print(f"最小值: {np.min(test_labels):.6f}")
    print(f"最大值: {np.max(test_labels):.6f}")
    print(f"均值: {np.mean(test_labels):.6f}")
    print(f"中位数: {np.median(test_labels):.6f}")
    print(f"标准差: {np.std(test_labels):.6f}")
    print(f"值域范围: {np.max(test_labels) - np.min(test_labels):.6f}")
    print(f"前10个真实值: {test_labels[:10].flatten()}")
    
    test_metrics = evaluate_predictions(test_labels, test_predictions, "测试数据")
    
    # 绘制预测vs真实值图表
    plt.figure(figsize=PREDICTION_PLOT_FIGSIZE)
    plt.scatter(test_labels, test_predictions, alpha=PREDICTION_SCATTER_ALPHA)
    plt.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'r--')
    plt.title('Test Data: Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.text(PREDICTION_TEXT_POS[0], PREDICTION_TEXT_POS[1], 
             f"R^2 = {test_metrics['r2']:.4f}\nRMSE = {test_metrics['rmse']:.4f}", 
             transform=plt.gca().transAxes, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(os.path.join(PREDICTION_RESULTS_DIR, 'test_prediction_results.png'), dpi=PREDICTION_PLOT_DPI)
    plt.close()
    
    # 保存包含真实值的预测结果
    results_df = pd.DataFrame({
        '真实值': test_labels.flatten(),
        '预测值': test_predictions.flatten(),
        '绝对误差': np.abs(test_labels.flatten() - test_predictions.flatten()),
        '相对误差': np.abs((test_labels.flatten() - test_predictions.flatten()) / (test_labels.flatten() + PREDICTION_DIV_ZERO_EPSILON)) * 100
    })
    
    # 添加绘制真实值和预测值的直方图
    plt.figure(figsize=PREDICTION_PLOT_FIGSIZE)
    plt.hist(test_labels.flatten(), bins=30, alpha=0.5, label='真实值')
    plt.hist(test_predictions.flatten(), bins=30, alpha=0.5, label='预测值')
    plt.title('真实值与预测值分布对比')
    plt.xlabel('值')
    plt.ylabel('频数')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PREDICTION_RESULTS_DIR, 'value_distribution.png'), dpi=PREDICTION_PLOT_DPI)
    plt.close()
    
    # 保存评估指标（修复之前误删的代码）
    metrics_df = pd.DataFrame({
        '指标': ['MSE', 'RMSE', 'MAE', 'R²', '平均相对误差(%)'],
        '值': [test_metrics['mse'], test_metrics['rmse'], test_metrics['mae'], 
               test_metrics['r2'], test_metrics['rel_error']]
    })
    metrics_df.to_csv(os.path.join(PREDICTION_RESULTS_DIR, 'test_evaluation_metrics.csv'), index=False)
    
    # 添加额外分析：检查预测值与真实值的相关性
    correlation = np.corrcoef(test_labels.flatten(), test_predictions.flatten())[0, 1]
    print(f"\n预测值与真实值的相关系数: {correlation:.6f}")
    
    # 检查是否有极端数值差异
    abs_diff = np.abs(test_labels.flatten() - test_predictions.flatten())
    max_diff_idx = np.argmax(abs_diff)
    print(f"最大绝对误差: {abs_diff[max_diff_idx]:.6f}, 位置: {max_diff_idx}")
    print(f"对应真实值: {test_labels.flatten()[max_diff_idx]:.6f}, 预测值: {test_predictions.flatten()[max_diff_idx]:.6f}")

else:
    # 只保存预测结果
    results_df = pd.DataFrame({
        '预测值': test_predictions.flatten()
    })

# 将结果保存到CSV文件
results_csv = os.path.join(PREDICTION_RESULTS_DIR, 'prediction_results.csv')
results_df.to_csv(results_csv, index=False)
print(f"\n预测结果已保存到: {results_csv}")

print("\n完成预测过程!")
print("="*80)