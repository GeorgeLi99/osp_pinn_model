"""
遗传算法优化器(Genetic Algorithm Optimizer)
用于PINN模型权重优化，替代传统梯度下降优化器
基于TensorFlow/Keras最新优化器API实现
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Optimizer


class GeneticOptimizer(tf.keras.optimizers.Optimizer):
    """基于遗传算法的优化器
    
    使用遗传算法原理（选择、交叉、变异）来更新神经网络权重，
    不依赖反向传播的梯度计算，可以跳出局部最优解。
    基于Keras 2.13+的优化器API实现。
    """
    
    def __init__(self, 
                 population_size=20,
                 mutation_rate=0.01,
                 crossover_rate=0.8,
                 selection_pressure=1.5,
                 learning_rate=0.01,
                 name="GeneticOptimizer", 
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
        # 确保将learning_rate作为位置参数传递给父类初始化方法
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        
        # 设置遗传算法特有的超参数
        self._population_size = population_size
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._selection_pressure = selection_pressure
        
        # 内部状态变量字典 - 存储每个变量的状态
        self._variable_states = {}
        self._iterations = tf.Variable(0, dtype=tf.int64, trainable=False)
        
    def build(self, var_list):
        """构建每个变量的状态"""
        super().build(var_list)
        # 基于传入的变量列表初始化状态字典
        for var in var_list:
            var_key = self._var_key(var)
            if var_key not in self._variable_states:
                # 初始化此变量的状态
                self._variable_states[var_key] = {
                    'population': None,  # 种群将在首次调用时初始化
                    'best_individual': tf.zeros_like(var, dtype=var.dtype),
                    'generation': tf.Variable(0, dtype=tf.int64, trainable=False),
                    'best_fitness': tf.Variable(-float('inf'), dtype=tf.float32, trainable=False)
                }
    
    def _var_key(self, var):
        """为变量创建唯一键"""
        return id(var)
        
    def _initialize_state_for_variable(self, variable):
        """初始化变量的状态"""
        return {
            'population': None,  # 种群将在首次调用时初始化
            'best_individual': tf.zeros_like(variable, dtype=variable.dtype),
            'generation': tf.Variable(0, dtype=tf.int64, trainable=False),
            'best_fitness': tf.Variable(-float('inf'), dtype=tf.float32, trainable=False)
        }
        
    def update_step(self, grad, variable, learning_rate=None):
        """实现遗传算法的权重更新逻辑
        
        Args:
            grad: 当前梯度 (Tensor).
            variable: 要更新的变量 (Tensor).
            learning_rate: 学习率，在遗传算法中我们使用内部学习率设置。
        """
        # 获取变量类型和键
        var_dtype = variable.dtype
        var_key = self._var_key(variable)
        
        # 检查并初始化状态（如果不存在）
        if var_key not in self._variable_states:
            self._variable_states[var_key] = self._initialize_state_for_variable(variable)
            
        state = self._variable_states[var_key]
        
        # 获取学习率
        if learning_rate is None:
            lr = self._get_hyper("learning_rate", variable.dtype)  # 使用_get_hyper获取学习率
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr = lr(self.iterations)
        else:
            lr = learning_rate
        population_size = tf.constant(self._population_size, dtype=tf.int32)  # 使用int32类型
        mutation_rate = tf.constant(self._mutation_rate, dtype=var_dtype)
        crossover_rate = tf.constant(self._crossover_rate, dtype=var_dtype)
        selection_pressure = tf.constant(self._selection_pressure, dtype=var_dtype)
        
        # 首次运行时初始化种群
        if state['population'] is None:
            # 创建随机种群，围绕当前权重的小范围变化
            rand_pop = tf.random.normal(
                shape=tf.concat([[population_size], tf.shape(variable)], axis=0),
                mean=0.0,
                stddev=0.1,
                dtype=var_dtype
            )
            # 将当前权重添加到随机种群
            state['population'] = tf.concat([tf.expand_dims(variable, 0), rand_pop[1:]], axis=0)
        
        # 计算每个个体的适应度（梯度的负方向指示提高适应度的方向）
        pop_flat = tf.reshape(state['population'], [population_size, -1])
        grad_flat = tf.reshape(grad, [-1])
        # 梯度的负方向是提高适应度的方向
        fitness = -tf.matmul(pop_flat, tf.expand_dims(grad_flat, 1))
        fitness = tf.reshape(fitness, [population_size])
        # 归一化适应度
        fitness = tf.nn.softmax(selection_pressure * fitness)
        
        # 选择操作 - 轮盘赌选择
        indices = tf.random.categorical(
            tf.math.log(tf.expand_dims(fitness, 0)), 
            population_size  # 该值已为int32类型
        )[0]
        selected_population = tf.gather(state['population'], indices)
        
        # 交叉操作
        # 随机配对父代
        indices = tf.random.shuffle(tf.range(population_size))
        half_pop = population_size // 2  # 使用整数除法
        parents1 = tf.gather(selected_population, indices[:half_pop])
        parents2 = tf.gather(selected_population, indices[half_pop:])
        
        # 随机交叉掩码
        mask = tf.cast(
            tf.random.uniform(
                tf.shape(parents1), 
                minval=0, 
                maxval=2, 
                dtype=tf.int32
            ), 
            dtype=var_dtype
        )
        
        # 执行交叉
        children1 = parents1 * mask + parents2 * (1 - mask)
        children2 = parents2 * mask + parents1 * (1 - mask)
        
        # 交叉概率掩码
        do_crossover = tf.random.uniform([half_pop], 0, 1) < crossover_rate
        do_crossover = tf.cast(do_crossover, var_dtype)
        do_crossover = tf.reshape(
            tf.repeat(
                tf.expand_dims(do_crossover, -1),
                repeats=tf.reduce_prod(tf.shape(parents1)[1:])
            ),
            tf.shape(parents1)
        )
        
        # 按交叉率决定是否使用交叉后的子代
        children1 = children1 * do_crossover + parents1 * (1 - do_crossover)
        children2 = children2 * do_crossover + parents2 * (1 - do_crossover)
        
        # 合并子代
        crossed_population = tf.concat([children1, children2], axis=0)
        
        # 变异操作
        mutation_values = tf.random.normal(
            tf.shape(crossed_population), 
            mean=0.0, 
            stddev=0.01, 
            dtype=var_dtype
        )
        
        # 变异掩码
        do_mutation = tf.random.uniform(tf.shape(crossed_population), 0, 1) < mutation_rate
        do_mutation = tf.cast(do_mutation, var_dtype)
        
        # 应用变异
        mutated_population = crossed_population + mutation_values * do_mutation
        
        # 获取当前种群中的最佳个体
        best_idx = tf.argmax(fitness)
        best_of_current = tf.gather(state['population'], best_idx)
        
        # 更新全局最佳个体
        current_best_fitness = tf.reduce_max(fitness)
        is_better = current_best_fitness > state['best_fitness']
        
        # 使用TensorFlow条件操作更新最佳个体和适应度
        # 注意：在图执行模式下使用tf.cond而不是Python的if语句
        def update_values():
            state['best_individual'].assign(best_of_current)
            state['best_fitness'].assign(current_best_fitness)
            return tf.constant(1.0)  # 返回任意值
            
        def keep_values():
            return tf.constant(0.0)  # 返回任意值
            
        _ = tf.cond(is_better, update_values, keep_values)
        
        # 精英保留 - 随机替换一个个体为当前最佳个体
        random_idx = tf.random.uniform([], 0, population_size, dtype=tf.int32)  # 显式指定为int32
        population_tensor = tf.tensor_scatter_nd_update(
            mutated_population,
            [[random_idx]],
            [best_of_current]
        )
        
        # 更新种群
        state['population'] = population_tensor
        state['generation'].assign_add(1)
        
        # 使用最佳个体和当前权重计算新的权重值
        new_value = variable + lr * (state['best_individual'] - variable)
        
        # 就地更新变量
        variable.assign(new_value)
        
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
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
        })
        return config
        
    def _serialize_hyperparameter(self, hyperparameter_name):
        """序列化超参数"""
        try:
            value = self._get_hyper(hyperparameter_name)
            if isinstance(value, tf.keras.optimizers.schedules.LearningRateSchedule):
                return tf.keras.optimizers.schedules.serialize(value)
            return value
        except AttributeError:
            # 兼容性处理，如果_get_hyper不可用
            return getattr(self, f"_{hyperparameter_name}", None)
        
    @classmethod
    def from_config(cls, config):
        """从配置创建实例"""
        if "lr" in config:
            config["learning_rate"] = config.pop("lr")
        if isinstance(config["learning_rate"], dict):
            config["learning_rate"] = tf.keras.optimizers.schedules.deserialize(config["learning_rate"])
        return cls(**config)
