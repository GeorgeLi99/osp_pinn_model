"""
使用Keras Tuner自动调整PINN模型超参数
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from config import *
from process_data import train_inputs, train_labels, val_inputs, val_labels
from simple_genetic_optimizer import SimpleGeneticOptimizer
from loss_pinn import PhysicsInformedLoss

# 检查GPU可用性
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 允许GPU内存按需增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, GPU_MEMORY_GROWTH)
        print(f"检测到 {len(gpus)} 个GPU设备，已配置内存增长设置")
    except RuntimeError as e:
        print(f"GPU配置错误: {e}")

# 创建模型构建函数，接受超参数
def build_model(hp):
    """
    构建可调参数的PINN模型
    
    Args:
        hp: Keras Tuner的超参数对象
    
    Returns:
        编译好的模型
    """
    # 确保使用正确的初始化器 - 保持用户自定义的初始化策略
    initializer = tf.keras.initializers.RandomNormal(
        mean=MODEL_WEIGHT_INIT_MEAN, 
        stddev=MODEL_WEIGHT_INIT_STDDEV
    )
    
    # 定义输入层
    inputs = tf.keras.layers.Input(shape=MODEL_INPUT_SHAPE, name="input_layer")
    
    # 构建动态层数和宽度的网络
    x = inputs
    
    # 第一层 - 可调单元数
    units_1 = hp.Int('units_1', min_value=16, max_value=64, step=8)
    dropout_1 = hp.Float('dropout_1', min_value=0.0, max_value=0.3, step=0.05)
    x = keras.layers.Dense(units_1, kernel_initializer=initializer, name="dense_1")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(dropout_1)(x)
    
    # 决定网络深度 (层数)
    num_layers = hp.Int('num_layers', 2, 6)
    
    # 动态添加更多层
    for i in range(2, num_layers + 1):
        # 每层神经元数量逐层减少
        max_units = max(8, int(units_1 / (i * 0.7)))
        min_units = max(4, int(units_1 / (i * 1.5)))
        
        units = hp.Int(f'units_{i}', min_value=min_units, max_value=max_units, step=4)
        dropout = hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.2, step=0.05)
        
        x = keras.layers.Dense(units, kernel_initializer=initializer, name=f"dense_{i}")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(dropout)(x)
    
    # 输出层
    outputs = keras.layers.Dense(
        MODEL_OUTPUT_UNITS, 
        activation=MODEL_OUTPUT_ACTIVATION, 
        kernel_initializer=initializer,
        name="output_layer"
    )(x)
    
    # 创建模型
    model = keras.Model(inputs=inputs, outputs=outputs, name="PINN_Tuned_Model")
    
    # 使用物理信息损失函数进行编译
    if USE_PINN_LOSS:
        loss_function = PhysicsInformedLoss(physics_weight=PINN_PHYSICS_WEIGHT)
    else:
        # 可调整损失函数
        loss_type = hp.Choice('loss_type', ['huber', 'mse', 'mae'])
        if loss_type == 'huber':
            loss_function = tf.keras.losses.Huber(delta=1.0)
        elif loss_type == 'mse':
            loss_function = tf.keras.losses.MeanSquaredError()
        else:
            loss_function = tf.keras.losses.MeanAbsoluteError()
    
    # 优化器选择 - 保留用户之前的设置
    if MODEL_OPTIMIZER.lower() == 'genetic':
        # 使用遗传算法优化器，调整一些参数
        population_size = hp.Int('population_size', min_value=20, max_value=60, step=10)
        mutation_rate = hp.Float('mutation_rate', min_value=0.01, max_value=0.05, step=0.01)
        
        optimizer = SimpleGeneticOptimizer(
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=OPTIMIZER_GENETIC_CROSSOVER_RATE,
            selection_pressure=OPTIMIZER_GENETIC_SELECTION_PRESSURE,
            learning_rate=OPTIMIZER_GENETIC_LEARNING_RATE
        )
    else:
        # 使用Adam优化器作为备选
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # 编译模型
    model.compile(
        loss=loss_function,
        optimizer=optimizer
    )
    
    return model

def run_hyperparameter_search():
    """执行超参数搜索"""
    # 创建超参数搜索对象 - 使用Hyperband算法
    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=30,  # 每个模型最多训练的轮数
        factor=3,       # Hyperband算法参数
        directory='tuner_results',
        project_name='pinn_model_tuning'
    )
    
    # 设置早停回调
    stop_early = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # 配置搜索
    print("\n开始超参数搜索...")
    tuner.search(
        train_inputs,
        train_labels,
        epochs=TRAINING_EPOCHS,
        validation_data=(val_inputs, val_labels),
        callbacks=[stop_early],
        verbose=1
    )
    
    # 获取最佳超参数
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n最佳超参数:")
    for param_name in best_hps.values:
        print(f"- {param_name}: {best_hps.values[param_name]}")
    
    # 获取最佳模型并再次训练到完整的轮数
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(
        train_inputs, 
        train_labels,
        epochs=TRAINING_EPOCHS,
        validation_data=(val_inputs, val_labels),
        verbose=1
    )
    
    # 保存最佳模型
    model_save_path = os.path.join(MODELS_DIR, 'best_tuned_model.h5')
    best_model.save(model_save_path)
    print(f"\n最佳模型已保存至: {model_save_path}")
    
    # 在验证集上评估模型
    eval_loss = best_model.evaluate(val_inputs, val_labels, verbose=0)
    print(f"\n最终验证损失: {eval_loss}")
    
    return best_model, best_hps

if __name__ == "__main__":
    # 首先检查依赖库是否已安装
    try:
        import keras_tuner
        print("keras_tuner已安装，开始超参数搜索...")
    except ImportError:
        print("未检测到keras_tuner库，正在尝试安装...")
        import subprocess
        subprocess.check_call(["pip", "install", "keras-tuner"])
        print("keras_tuner安装完成，开始超参数搜索...")
    
    # 运行超参数搜索
    best_model, best_hps = run_hyperparameter_search()
