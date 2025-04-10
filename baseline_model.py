"""
Baseline Model for PINN Model Comparison
This script implements a simple baseline model using TensorFlow and Keras that predicts 
the mean of the training data for all inputs, regardless of feature values.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
# 设置matplotlib使用非交互式后端，避免图形显示问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 尝试设置中文字体
try:
    # 尝试设置简单的sans-serif字体，避免中文渲染问题
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except Exception as e:
    print(f"配置字体时出错: {e}")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from config import *

class MeanBaselineModel:
    """
    A simple baseline model that predicts the mean of the training data
    regardless of the input features.
    """
    def __init__(self):
        self.mean_value = None
        self.model = None
        
    def build_model(self, input_shape):
        """
        Build a Keras model that always outputs the mean value
        regardless of the input.
        
        Args:
            input_shape: Shape of input features
        """
        # Create a model that takes inputs but always predicts the same value
        inputs = tf.keras.Input(shape=input_shape)
        # Using a Lambda layer to output a constant value
        outputs = tf.keras.layers.Lambda(lambda x: tf.ones_like(x[:, :1]) * self.mean_value)(inputs)
        
        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model with MSE loss (doesn't really matter as weights won't change)
        self.model.compile(optimizer='adam', loss='mse')
        
        return self.model
    
    def fit(self, X_train, y_train, verbose=1):
        """
        'Fit' the baseline model by computing the mean of training data.
        
        Args:
            X_train: Training features
            y_train: Training target values
            verbose: Verbosity level
            
        Returns:
            Dictionary containing loss history
        """
        # Calculate mean of training labels
        self.mean_value = np.mean(y_train)
        
        if verbose:
            print(f"训练集目标值的平均值: {self.mean_value:.8f}")
        
        # Build the model with the mean value
        input_shape = (X_train.shape[1],)
        self.build_model(input_shape)
        
        # Calculate losses for return (history-like dict)
        train_predictions = np.full_like(y_train, self.mean_value)
        train_loss = mean_squared_error(y_train, train_predictions)
        
        # Create a dict similar to Keras history object
        history = {
            'loss': [train_loss],  # Just one epoch since we don't actually train
            'val_loss': [train_loss]  # Same as training loss for simplicity
        }
        
        if verbose:
            print(f"训练集损失 (MSE): {train_loss:.8f}")
        
        return history
    
    def predict(self, X):
        """
        Make predictions using the baseline model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions (all equal to the mean value)
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        return self.model.predict(X)

def compute_cost(y_pred, y_true):
    """
    Compute MSE between predictions and true values.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        
    Returns:
        MSE value
    """
    return mean_squared_error(y_true, y_pred)

def evaluate_baseline(baseline_pred, y_true, name="测试集"):
    """
    Evaluate baseline model predictions with multiple metrics.
    
    Args:
        baseline_pred: Baseline model predictions
        y_true: True values
        name: Name of the dataset for reporting
    """
    mse = mean_squared_error(y_true, baseline_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, baseline_pred)
    r2 = r2_score(y_true, baseline_pred)
    
    print(f"基线模型在{name}上的评估结果:")
    print(f"MSE: {mse:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"MAE: {mae:.8f}")
    print(f"R²: {r2:.8f}")
    
    return mse, rmse, mae, r2

def plot_losses(history, save_path=None):
    """
    Plot training and validation loss.
    
    Args:
        history: Dictionary containing loss values (similar to Keras history object)
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=PREDICTION_PLOT_FIGSIZE, dpi=PREDICTION_PLOT_DPI)
    
    # For the baseline model, we only have one epoch, so we'll create a simple bar chart
    # or extend the history to show the constant loss value over multiple epochs for visualization
    
    # Extend single loss value to multiple epochs for better visualization
    epochs = 5  # Arbitrary number of epochs for visualization
    train_loss = history['loss'][0]
    val_loss = history.get('val_loss', [train_loss])[0]
    
    extended_epochs = list(range(1, epochs + 1))
    extended_train_loss = [train_loss] * epochs
    extended_val_loss = [val_loss] * epochs
    
    # Plot the losses
    plt.plot(extended_epochs, extended_train_loss, 'b-', label='Training Loss')
    plt.plot(extended_epochs, extended_val_loss, 'r-', label='Validation Loss')
    
    # Add a horizontal line for the baseline MSE value
    plt.axhline(y=train_loss, color='gray', linestyle='--', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Baseline Model Loss (Mean Prediction)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Annotate the constant loss value
    plt.annotate(f'Constant Loss: {train_loss:.6f}', 
                 xy=(epochs/2, train_loss), 
                 xytext=(epochs/2, train_loss * 1.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 ha='center')
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to: {save_path}")
    
    # Close the figure to prevent display
    plt.close()

def plot_baseline_comparison(y_true, baseline_pred, pinn_pred=None, save_path=None):
    """
    Plot comparison between true values, baseline predictions, and PINN predictions.
    
    Args:
        y_true: True values
        baseline_pred: Baseline model predictions
        pinn_pred: PINN model predictions (optional)
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=PREDICTION_PLOT_FIGSIZE, dpi=PREDICTION_PLOT_DPI)
    
    # Plot points
    plt.scatter(range(len(y_true)), y_true, alpha=0.7, label='Real Values', color='blue')
    plt.scatter(range(len(baseline_pred)), baseline_pred, alpha=0.7, label='Baseline Predictions', color='red')
    
    if pinn_pred is not None:
        plt.scatter(range(len(pinn_pred)), pinn_pred, alpha=0.7, label='PINN Predictions', color='green')
    
    # Add lines for better visualization
    plt.plot(range(len(y_true)), y_true, alpha=0.3, color='blue')
    plt.plot(range(len(baseline_pred)), baseline_pred, alpha=0.3, color='red')
    
    if pinn_pred is not None:
        plt.plot(range(len(pinn_pred)), pinn_pred, alpha=0.3, color='green')
    
    # Add labels and title (using English to avoid font issues)
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.title('Baseline Model vs. Real Values')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    
    # Close the figure to prevent display
    plt.close()

# 为计算MSE定义额外的辅助函数
def print_baseline_performance(y_true, baseline_pred):
    """打印基线模型性能指标"""
    print(f"\n基线模型 (预测平均值) 在测试集上的评估:")
    mse = mean_squared_error(y_true, baseline_pred)
    mae = mean_absolute_error(y_true, baseline_pred)
    r2 = r2_score(y_true, baseline_pred)
    print(f"MSE: {mse:.8f}")
    print(f"MAE: {mae:.8f}")
    print(f"R²: {r2:.8f}")
    return mse, mae, r2

if __name__ == "__main__":
    print("="*80)
    print("基线模型评估")
    print("="*80)
    
    # 读取训练数据
    try:
        print(f"读取训练数据: {DATA_PATH}")
        data = pd.read_csv(DATA_PATH, header=0)
        print(f"训练数据形状: {data.shape}")
        
        # 提取特征和标签
        if isinstance(COLUMN, int):
            features_df = data.iloc[:, :COLUMN]
            labels_df = data.iloc[:, COLUMN]
        else:
            # 如果COLUMN是列名
            columns = data.columns.tolist()
            if COLUMN in columns:
                col_idx = columns.index(COLUMN)
                features_df = data.iloc[:, :col_idx]
                labels_df = data[COLUMN]
            else:
                # 假设最后一列是标签
                features_df = data.iloc[:, :-1]
                labels_df = data.iloc[:, -1]
        
        # 转换为numpy数组
        X = features_df.values.astype(np.float32)
        y = labels_df.values.astype(np.float32)
        
        # 划分训练集和测试集
        indices_permutation = np.random.permutation(len(y))
        train_size = int(TRAIN_TEST_SPLIT * len(y))
        
        train_indices = indices_permutation[:train_size]
        test_indices = indices_permutation[train_size:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
        print(f"测试集形状: X={X_test.shape}, y={y_test.shape}")
        
        # 1. 创建和训练基线模型
        print("\n创建并训练基线模型 (预测平均值)...")
        baseline_model = MeanBaselineModel()
        history = baseline_model.fit(X_train, y_train)
        
        # 2. 在测试集上进行预测
        baseline_predictions = baseline_model.predict(X_test)
        
        # 3. 评估基线模型性能
        print("\n基线模型评估结果:")
        baseline_mse, baseline_rmse, baseline_mae, baseline_r2 = evaluate_baseline(
            baseline_predictions, y_test
        )
        
        # 4. 创建比较图表
        try:
            os.makedirs(PREDICTION_RESULTS_DIR, exist_ok=True)
            # 预测值与真实值比较图
            plot_path = os.path.join(PREDICTION_RESULTS_DIR, 'baseline_comparison.png')
            # 限制样本数量，避免图表过大
            max_samples = min(1000, len(y_test))
            plot_baseline_comparison(y_test[:max_samples], baseline_predictions[:max_samples], save_path=plot_path)
            
            # 损失图表
            loss_plot_path = os.path.join(PREDICTION_RESULTS_DIR, 'baseline_loss.png')
            plot_losses(history, save_path=loss_plot_path)
        except Exception as e:
            print(f"创建图表时出错: {e}")
        
        # 5. 计算并显示训练集与测试集损失比较
        train_predictions = np.full_like(y_train, baseline_model.mean_value)
        train_loss = mean_squared_error(y_train, train_predictions)
        test_loss = baseline_mse  # 与前面计算的MSE相同
        
        print(f"\n损失比较:")
        print(f"训练集损失 (MSE): {train_loss:.8f}")
        print(f"测试集损失 (MSE): {test_loss:.8f}")
        print(f"差异: {test_loss - train_loss:.8f} ({(test_loss - train_loss) * 100 / train_loss:.2f}%)")
        
        # 6. 打印额外信息
        print("\n基线模型信息:")
        print(f"训练样本数: {X_train.shape[0]}")
        print(f"测试样本数: {X_test.shape[0]}")
        print(f"特征数量: {X_train.shape[1]}")
        print(f"预测常数值: {baseline_model.mean_value:.8f}")
        
        # 如果要在测试数据上运行
        try:
            # 检查测试数据文件是否存在
            if os.path.exists(TEST_DATA_PATH):
                print(f"\n在单独的测试数据上评估基线模型: {TEST_DATA_PATH}")
                test_data = pd.read_csv(TEST_DATA_PATH)
                
                # 确定要使用的列
                if TEST_DATA_COLUMN is not None:
                    target_column = TEST_DATA_COLUMN
                else:
                    target_column = COLUMN
                
                # 提取特征
                if isinstance(target_column, int):
                    test_features = test_data.iloc[:, :target_column].values.astype(np.float32)
                    if target_column < test_data.shape[1]:  # 如果目标列存在
                        test_labels = test_data.iloc[:, target_column].values.astype(np.float32)
                        has_labels = True
                    else:
                        has_labels = False
                else:
                    # 假设target_column是列名
                    if target_column in test_data.columns:
                        col_idx = test_data.columns.tolist().index(target_column)
                        test_features = test_data.iloc[:, :col_idx].values.astype(np.float32)
                        test_labels = test_data[target_column].values.astype(np.float32)
                        has_labels = True
                    else:
                        # 假设没有标签列，全部都是特征
                        test_features = test_data.values.astype(np.float32)
                        has_labels = False
                
                # 确保特征列数与训练数据相同
                if test_features.shape[1] != X_train.shape[1]:
                    print(f"警告: 测试数据特征数 ({test_features.shape[1]}) 与训练数据特征数 ({X_train.shape[1]}) 不匹配")
                    print("尝试调整测试数据特征...")
                    
                    if test_features.shape[1] > X_train.shape[1]:
                        # 如果测试数据特征多，截取前面的列
                        test_features = test_features[:, :X_train.shape[1]]
                    else:
                        # 如果测试数据特征少，无法预测
                        print("错误: 测试数据特征不足，无法进行预测")
                        raise ValueError("测试数据特征不足")
                
                # 使用基线模型进行预测
                test_predictions = baseline_model.predict(test_features)
                
                # 如果有标签，评估性能
                if has_labels:
                    print("\n在测试数据上的基线模型评估结果:")
                    test_mse, test_rmse, test_mae, test_r2 = evaluate_baseline(
                        test_predictions, test_labels, name="外部测试集"
                    )
                    
                    # 创建比较图表
                    try:
                        test_plot_path = os.path.join(PREDICTION_RESULTS_DIR, 'baseline_test_comparison.png')
                        # 限制样本数量，避免图表过大
                        max_samples = min(1000, len(test_labels))
                        plot_baseline_comparison(test_labels[:max_samples], test_predictions[:max_samples], save_path=test_plot_path)
                        
                        # 创建损失比较图 - 包含外部测试集的结果
                        external_test_loss = mean_squared_error(test_labels, test_predictions)
                        combined_history = {
                            'loss': [train_loss],
                            'val_loss': [baseline_mse],
                            'test_loss': [external_test_loss]
                        }
                        test_loss_plot_path = os.path.join(PREDICTION_RESULTS_DIR, 'baseline_loss_with_external_test.png')
                        plot_losses(combined_history, save_path=test_loss_plot_path)
                        
                        print(f"\n外部测试集损失比较:")
                        print(f"训练集损失 (MSE): {train_loss:.8f}")
                        print(f"验证集损失 (MSE): {baseline_mse:.8f}")
                        print(f"外部测试集损失 (MSE): {external_test_loss:.8f}")
                    except Exception as e:
                        print(f"创建测试数据图表时出错: {e}")
                else:
                    print("测试数据没有标签列，无法评估性能")
                    
                # 保存测试数据的预测结果
                test_pred_df = pd.DataFrame({
                    'Baseline_Prediction': test_predictions.flatten()
                })
                
                test_pred_path = os.path.join(PREDICTION_RESULTS_DIR, 'baseline_test_predictions.csv')
                test_pred_df.to_csv(test_pred_path, index=False)
                print(f"测试数据预测结果已保存至: {test_pred_path}")
        
        except Exception as e:
            print(f"在测试数据上评估时出错: {e}")
    
    except Exception as e:
        print(f"运行基线模型时出错: {e}")
