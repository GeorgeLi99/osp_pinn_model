"""
基于物理约束的神经网络损失函数
实现物理信息神经网络(Physics-Informed Neural Networks, PINN)核心组件
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from config import PINN_PHYSICS_WEIGHT, PINN_SMOOTHNESS_WEIGHT, PINN_NEGATIVE_PENALTY_WEIGHT

# 尝试获取适用于当前TensorFlow版本的序列化装饰器
try:
    # TF 2.4+
    from tensorflow.keras.utils import register_keras_serializable
except ImportError:
    try:
        # 尝试其他可能的路径
        from tensorflow.python.keras.utils.generic_utils import register_keras_serializable
    except ImportError:
        # 如果无法导入，创建一个空装饰器
        def register_keras_serializable(package=None, name=None):
            def decorator(cls):
                return cls
            return decorator

@register_keras_serializable(package="loss_pinn")
class PhysicsInformedLoss(keras.losses.Loss):
    """
    物理信息神经网络的基础损失函数
    
    将物理约束直接融入损失函数中，使神经网络学习符合物理规律的解
    """
    def __init__(self, physics_weight=PINN_PHYSICS_WEIGHT, name="physics_informed_loss"):
        super().__init__(name=name)
        self.physics_weight = physics_weight
        # 使用MSE作为数据损失
        self.data_loss_fn = keras.losses.MeanSquaredError()
        
    def call(self, y_true, y_pred):
        # 数据拟合损失
        data_loss = self.data_loss_fn(y_true, y_pred)
        
        # 物理约束损失
        physics_loss = self.physics_constraint(y_true, y_pred)
        
        # 总损失 = 数据损失 + 物理约束权重 * 物理约束损失
        return data_loss + self.physics_weight * physics_loss
    
    def physics_constraint(self, y_true, y_pred):
        """
        实现物理约束项 - 避免使用if语句的版本
        
        在图执行模式下安全的版本，不使用Python条件语句
        """
        # 获取y_pred的数据类型，用于创建匹配的常量
        dtype = y_pred.dtype
        
        # 非负约束 - 对负值进行惩罚
        # 使用与y_pred相同数据类型的常量0.0
        zero = tf.constant(0.0, dtype=dtype)
        negative_penalty = tf.reduce_mean(tf.square(tf.maximum(zero, -y_pred)))
        
        # 平滑性约束 - 无需条件判断
        # 所有情况都计算平滑性惩罚，无效情况结果为0
        
        # 安全地计算平滑度约束，避免使用if语句
        # 将y_pred展平为1D张量
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        # 计算相邻元素的差异
        # 安全地切片 - 即使长度为1也不会出错，会得到空张量
        y_head = y_pred_flat[1:]  # 从第二个元素到最后
        y_tail = y_pred_flat[:-1]  # 从第一个元素到倒数第二个
        
        # 确保有元素可以计算差异
        # 如果y_pred_flat的长度至少为2，则diff会有至少1个元素
        diff = y_head - y_tail
        
        # 计算平方差的均值
        # 如果diff是空张量，reduce_mean会返回NaN，我们使用tf.cond安全处理
        # 检查diff是否为空
        diff_size = tf.size(diff)
        
        # 使用tf.cond检查diff_size是否大于0，确保常量匹配数据类型
        smoothness_penalty = tf.cond(
            diff_size > 0,
            lambda: tf.reduce_mean(tf.square(diff)),
            lambda: tf.constant(0.0, dtype=dtype)  # 使用与y_pred相同的数据类型
        )
        
        # 返回总的物理约束损失，使用配置中的权重
        negative_weight = tf.constant(PINN_NEGATIVE_PENALTY_WEIGHT, dtype=dtype)
        smoothness_weight = tf.constant(PINN_SMOOTHNESS_WEIGHT, dtype=dtype)
        return negative_weight * negative_penalty + smoothness_weight * smoothness_penalty
    
    # 添加get_config方法以支持序列化
    def get_config(self):
        config = super().get_config()
        config.update({"physics_weight": self.physics_weight})
        return config


def compute_derivatives(model, x):
    """
    计算模型输出关于输入的导数
    
    参数：
    model: Keras模型
    x: 输入数据
    
    返回：
    元组: (y_pred, dy_dx, d2y_dx2)
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    
    # 确保x可以追踪梯度
    x = tf.Variable(x)
    
    # 使用自动微分计算导数
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            y_pred = model(x)
            
        # 计算一阶导数
        dy_dx = tape1.gradient(y_pred, x)
        if dy_dx is None:
            dy_dx = tf.zeros_like(x)
    
    # 计算二阶导数
    d2y_dx2 = tape2.gradient(dy_dx, x)
    if d2y_dx2 is None:
        d2y_dx2 = tf.zeros_like(x)
    
    # 清理
    del tape1, tape2
    
    return y_pred, dy_dx, d2y_dx2


class PDELoss(PhysicsInformedLoss):
    """
    基于偏微分方程(PDE)的损失函数
    
    用于解决由偏微分方程定义的物理问题
    """
    def __init__(self, pde_weight=1.0, bc_weight=1.0, ic_weight=1.0, name="pde_loss"):
        super().__init__(physics_weight=1.0, name=name)
        self.pde_weight = pde_weight  # PDE残差权重
        self.bc_weight = bc_weight    # 边界条件权重
        self.ic_weight = ic_weight    # 初始条件权重
    
    def physics_constraint(self, y_true, y_pred):
        """
        实现PDE约束
        
        子类应该重写此方法来实现特定的PDE约束
        """
        # 这是一个占位符，实际应用中需要根据具体PDE重写
        return tf.constant(0.0)
    
    def boundary_constraint(self, x_bc, y_bc=None):
        """
        边界条件约束
        
        参数:
        x_bc: 边界点的坐标
        y_bc: 边界点的真实值（如果有）
        
        返回：
        边界条件损失
        """
        # 这是一个占位符，实际应用中需要根据具体问题重写
        return tf.constant(0.0)
    
    def initial_constraint(self, x_ic, y_ic):
        """
        初始条件约束
        
        参数:
        x_ic: 初始点的坐标
        y_ic: 初始点的真实值
        
        返回：
        初始条件损失
        """
        # 这是一个占位符，实际应用中需要根据具体问题重写
        return tf.constant(0.0)


class WaveEquationLoss(PDELoss):
    """
    波动方程损失函数
    
    用于解决波动方程问题: ∂²u/∂t² - c²∂²u/∂x² = 0
    """
    def __init__(self, wave_speed=1.0, name="wave_equation_loss"):
        super().__init__(name=name)
        self.wave_speed = wave_speed  # 波速
    
    def physics_constraint(self, y_true, y_pred):
        """
        波动方程约束: ∂²u/∂t² - c²∂²u/∂x² = 0
        
        使用自动微分计算残差
        """
        # 假设输入x的格式为 [x_position, time]
        # 这里简化处理，实际应用中需要根据具体问题调整
        
        # 残差计算示例（简化版）
        # 在实际应用中，需要计算真正的波动方程残差
        _, _, d2y_dx2 = compute_derivatives(self.model, self.x)
        residual = d2y_dx2 + self.wave_speed * tf.square(y_pred)
        
        return tf.reduce_mean(tf.square(residual))


class DerivativeConstrainedLoss(PhysicsInformedLoss):
    """
    基于导数约束的损失函数
    
    对预测结果的导数施加约束，适用于需要满足特定导数行为的问题
    """
    def __init__(self, derivative_weight=0.5, name="derivative_constrained_loss"):
        super().__init__(physics_weight=derivative_weight, name=name)
        
    def physics_constraint(self, y_true, y_pred):
        """
        导数约束实现
        
        约束模型预测的导数行为
        """
        # 获取输入变量，这里假设模型记录了训练输入
        if hasattr(self, 'current_inputs'):
            x = self.current_inputs
        else:
            # 如果没有记录输入，则返回一个基本约束
            return super().physics_constraint(y_true, y_pred)
        
        # 计算预测值的导数
        _, dy_dx, _ = compute_derivatives(self.model, x)
        
        # 约束导数的行为，例如限制导数的大小
        derivative_constraint = tf.reduce_mean(tf.square(dy_dx))
        
        return derivative_constraint


class PhysicsConstrainedOptimizer(tf.keras.optimizers.Optimizer):
    """
    物理约束优化器
    
    在优化过程中直接应用物理约束，通过修改梯度来强制满足物理规律
    """
    def __init__(self, 
                 base_optimizer=None,
                 physics_weight=0.1,
                 name="PhysicsConstrainedOptimizer",
                 **kwargs):
        super().__init__(name, **kwargs)
        
        # 如果没有提供优化器，默认使用Adam
        if base_optimizer is None:
            self.base_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        else:
            self.base_optimizer = base_optimizer
        
        self.physics_weight = physics_weight
    
    def _create_slots(self, var_list):
        self.base_optimizer._create_slots(var_list)
    
    def _prepare_local(self, var_device, var_dtype, apply_state):
        self.base_optimizer._prepare_local(var_device, var_dtype, apply_state)
    
    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        # 在应用梯度前，可以根据物理约束修改梯度
        modified_grads_and_vars = []
        
        for grad, var in grads_and_vars:
            if grad is not None:
                # 这里可以添加基于物理约束的梯度修正
                modified_grads_and_vars.append((grad, var))
            else:
                modified_grads_and_vars.append((grad, var))
        
        return self.base_optimizer.apply_gradients(modified_grads_and_vars, name=name, **kwargs)
    
    def get_config(self):
        config = self.base_optimizer.get_config()
        config.update({
            'physics_weight': self.physics_weight,
        })
        return config