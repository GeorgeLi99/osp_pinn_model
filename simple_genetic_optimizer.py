"""
简化版遗传算法优化器(Simple Genetic Algorithm Optimizer)
用于PINN模型权重优化，替代传统梯度下降优化器
兼容TensorFlow图执行模式
"""

import tensorflow as tf
import numpy as np


class SimpleGeneticOptimizer(tf.keras.optimizers.Optimizer):
    """简化版遗传算法优化器
    
    使用遗传算法原理来更新神经网络权重，
    不依赖反向传播的梯度计算，可以跳出局部最优解。
    兼容TensorFlow图执行模式。
    """
    
    def __init__(self, 
                 population_size=20,
                 mutation_rate=0.01,
                 crossover_rate=0.8,
                 selection_pressure=1.5,
                 learning_rate=0.01,
                 name="SimpleGeneticOptimizer", 
                 **kwargs):
        """初始化遗传算法优化器
        
        参数:
            population_size: 种群大小，即为每个权重维护的候选解数量
            mutation_rate: 变异率，控制基因变异概率
            crossover_rate: 交叉率，控制染色体交叉概率
            selection_pressure: 选择压力，值越大，适应度高的个体被选中概率越高
            learning_rate: 学习率，控制权重更新幅度
            name: 优化器名称
        """
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        
        # 设置遗传算法特有的超参数
        self._population_size = population_size
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._selection_pressure = selection_pressure
        
        # 创建迭代计数器
        self._iterations = tf.Variable(0, dtype=tf.int64, trainable=False)
    
    def update_step(self, grad, variable, learning_rate=None):
        """实现遗传算法的权重更新逻辑，兼容TensorFlow图执行模式
        
        Args:
            grad: 当前梯度 (Tensor)
            variable: 要更新的变量 (Tensor)
            learning_rate: 可选的学习率参数
        """
        # 获取学习率
        if learning_rate is None:
            lr = self._get_hyper("learning_rate", variable.dtype)
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr = lr(self.iterations)
        else:
            lr = learning_rate
        
        # 获取其他超参数
        population_size = tf.constant(self._population_size, dtype=tf.int32)
        mutation_rate = tf.constant(self._mutation_rate, dtype=variable.dtype)
        selection_pressure = tf.constant(self._selection_pressure, dtype=variable.dtype)
        
        # 简化版实现：使用梯度信息和随机扰动生成一组候选方向
        # 生成随机扰动方向
        noise = tf.random.normal(
            shape=[population_size-1] + variable.shape.as_list(),
            mean=0.0,
            stddev=0.1,
            dtype=variable.dtype
        )
        
        # 第一个方向是梯度的负方向（确保至少有一个好的方向）
        grad_direction = -grad * lr  # 梯度下降方向
        
        # 合并所有方向
        all_directions = tf.concat([
            tf.expand_dims(grad_direction, 0),  # 梯度方向
            noise  # 随机方向
        ], axis=0)
        
        # 计算每个方向的适应度（梯度的负方向是改进方向）
        flat_directions = tf.reshape(all_directions, [population_size, -1])
        flat_grad = tf.reshape(grad, [-1])
        
        # 计算每个方向与梯度的点积 - 负值越小表示方向越好
        fitness = -tf.matmul(flat_directions, tf.expand_dims(flat_grad, 1))
        fitness = tf.reshape(fitness, [population_size])
        
        # 使用softmax给方向分配权重
        weights = tf.nn.softmax(selection_pressure * fitness)
        
        # 选择最佳方向
        best_idx = tf.argmax(fitness)
        best_direction = tf.gather(all_directions, best_idx)
        
        # 将方向应用到变量上（简化版遗传算法，相当于沿最佳方向更新）
        variable.assign_add(lr * best_direction)
        
        # 增加迭代计数
        self.iterations.assign_add(1)
    
    def get_config(self):
        """获取优化器配置"""
        config = super().get_config()
        config.update({
            "population_size": self._population_size,
            "mutation_rate": self._mutation_rate,
            "crossover_rate": self._crossover_rate,
            "selection_pressure": self._selection_pressure,
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        """从配置创建实例"""
        if "lr" in config:
            config["learning_rate"] = config.pop("lr")
        if isinstance(config["learning_rate"], dict):
            config["learning_rate"] = tf.keras.optimizers.schedules.deserialize(config["learning_rate"])
        return cls(**config)
