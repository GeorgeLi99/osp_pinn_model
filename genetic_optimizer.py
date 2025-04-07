"""
遗传算法优化器(Genetic Algorithm Optimizer)
用于PINN模型权重优化，替代传统梯度下降优化器
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


class GeneticOptimizer(tf.keras.optimizers.Optimizer):
    """基于遗传算法的优化器
    
    使用遗传算法原理（选择、交叉、变异）来更新神经网络权重，
    不依赖反向传播的梯度计算，可以跳出局部最优解。
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
        super(GeneticOptimizer, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("population_size", population_size)
        self._set_hyper("mutation_rate", mutation_rate)
        self._set_hyper("crossover_rate", crossover_rate)
        self._set_hyper("selection_pressure", selection_pressure)
        self._populations = {}
        self._fitnesses = {}
        self._iterations = 0
        
    def _create_slots(self, var_list):
        """为变量创建额外的插槽（优化器状态）"""
        for var in var_list:
            # 为每个变量创建一个种群
            self.add_slot(var, "population", initializer=tf.zeros_like(var, dtype=var.dtype))
            # 创建最佳个体插槽
            self.add_slot(var, "best_individual", initializer=tf.zeros_like(var, dtype=var.dtype))
            # 创建当前代数插槽
            self.add_slot(var, "generation", initializer=tf.zeros(shape=(), dtype=tf.int64))
            # 创建最佳适应度插槽
            self.add_slot(var, "best_fitness", initializer=tf.constant(-float('inf'), dtype=tf.float32))
    
    def _resource_apply_dense(self, grad, var, apply_state=None):
        """使用遗传算法更新变量的稠密实现"""
        var_device, var_dtype = var.device, var.dtype.base_dtype
        
        # 获取超参数
        lr = self._get_hyper("learning_rate", var_dtype)
        population_size = self._get_hyper("population_size", tf.int64)
        mutation_rate = self._get_hyper("mutation_rate", var_dtype)
        crossover_rate = self._get_hyper("crossover_rate", var_dtype)
        selection_pressure = self._get_hyper("selection_pressure", var_dtype)
        
        # 获取插槽
        population = self.get_slot(var, "population")
        best_individual = self.get_slot(var, "best_individual")
        generation = self.get_slot(var, "generation")
        best_fitness = self.get_slot(var, "best_fitness")
        
        # 首次运行时初始化种群
        def initialize_population():
            # 创建随机种群，围绕当前权重的小范围变化
            rand_pop = tf.random.normal(
                shape=tf.concat([[population_size], tf.shape(var)], axis=0),
                mean=0.0,
                stddev=0.1,
                dtype=var_dtype
            )
            # 将当前权重作为第一个个体
            pop = var + rand_pop
            return pop
        
        # 如果是第一代，初始化种群
        population_init = tf.cond(
            tf.equal(generation, 0),
            lambda: initialize_population(),
            lambda: population
        )
        
        # 计算每个个体的适应度（这里使用梯度的负值作为适应度的一部分）
        # 在实际GA中，适应度通常是通过目标函数计算的，这里我们用梯度的方向作为适应度指导
        def calculate_fitness(pop):
            # 展开种群为一维列表，计算每个个体与当前梯度的点积
            pop_flat = tf.reshape(pop, [population_size, -1])
            grad_flat = tf.reshape(grad, [-1])
            # 梯度的负方向是提高适应度的方向
            fitness = -tf.matmul(pop_flat, tf.expand_dims(grad_flat, 1))
            # 归一化适应度以避免数值问题
            fitness = tf.nn.softmax(selection_pressure * fitness)
            return tf.reshape(fitness, [population_size])
        
        fitness = calculate_fitness(population_init)
        
        # 选择操作 - 轮盘赌选择
        def selection(pop, fit):
            # 使用适应度作为选择概率
            indices = tf.random.categorical(
                tf.math.log(tf.expand_dims(fit, 0)), 
                population_size
            )[0]
            selected = tf.gather(pop, indices)
            return selected
        
        selected_population = selection(population_init, fitness)
        
        # 交叉操作
        def crossover(pop):
            # 随机配对父代
            indices = tf.random.shuffle(tf.range(population_size))
            parents1 = tf.gather(pop, indices[:population_size//2])
            parents2 = tf.gather(pop, indices[population_size//2:])
            
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
            
            # 交叉概率掩码
            crossover_mask = tf.cast(
                tf.random.uniform(
                    [population_size//2], 
                    minval=0, 
                    maxval=1, 
                    dtype=var_dtype
                ) < crossover_rate, 
                dtype=var_dtype
            )
            crossover_mask = tf.reshape(
                tf.tile(
                    tf.expand_dims(crossover_mask, -1), 
                    [1, tf.reduce_prod(tf.shape(parents1)[1:])]
                ),
                tf.shape(parents1)
            )
            
            # 执行交叉
            children1 = parents1 * mask + parents2 * (1 - mask)
            children2 = parents2 * mask + parents1 * (1 - mask)
            
            # 根据交叉率决定是否使用交叉后的子代
            children1 = children1 * crossover_mask + parents1 * (1 - crossover_mask)
            children2 = children2 * crossover_mask + parents2 * (1 - crossover_mask)
            
            # 合并子代
            children = tf.concat([children1, children2], axis=0)
            return children
        
        crossed_population = crossover(selected_population)
        
        # 变异操作
        def mutation(pop):
            # 生成随机变异
            mutation_values = tf.random.normal(
                tf.shape(pop), 
                mean=0.0, 
                stddev=0.01, 
                dtype=var_dtype
            )
            
            # 变异掩码
            mutation_mask = tf.cast(
                tf.random.uniform(
                    tf.shape(pop), 
                    minval=0, 
                    maxval=1, 
                    dtype=var_dtype
                ) < mutation_rate, 
                dtype=var_dtype
            )
            
            # 应用变异
            mutated = pop + mutation_values * mutation_mask
            return mutated
        
        mutated_population = mutation(crossed_population)
        
        # 精英保留：用最佳个体替换一个随机个体
        best_idx = tf.argmax(fitness)
        best_of_current = tf.gather(population_init, best_idx)
        
        # 更新全局最佳个体
        def update_best():
            current_best_fitness = tf.reduce_max(fitness)
            is_better = current_best_fitness > best_fitness
            new_best_fitness = tf.where(
                is_better, 
                current_best_fitness, 
                best_fitness
            )
            new_best_individual = tf.where(
                is_better, 
                best_of_current, 
                best_individual
            )
            return new_best_fitness, new_best_individual
        
        new_best_fitness, new_best_individual = update_best()
        
        # 精英保留，确保最佳个体不会丢失
        random_idx = tf.random.uniform([], minval=0, maxval=population_size, dtype=tf.int32)
        elitist_indices = tf.range(population_size, dtype=tf.int32)
        elitist_mask = tf.equal(elitist_indices, random_idx)
        elitist_indices = tf.where(elitist_mask, best_idx, elitist_indices)
        final_population = tf.tensor_scatter_nd_update(
            mutated_population,
            tf.expand_dims(tf.expand_dims(random_idx, -1), -1),
            tf.expand_dims(best_of_current, 0)
        )
        
        # 从种群中计算新的权重（取种群平均值向最佳个体偏移）
        new_var = var + lr * (new_best_individual - var)
        
        # 更新插槽和变量
        with tf.control_dependencies([
            population.assign(final_population),
            best_individual.assign(new_best_individual),
            best_fitness.assign(new_best_fitness),
            generation.assign_add(1)
        ]):
            return var.assign(new_var)
    
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        """稀疏梯度更新实现（简化版，实际上调用稠密实现）"""
        # 将稀疏梯度转换为稠密梯度
        dense_grad = tf.zeros_like(var)
        dense_grad = tf.tensor_scatter_nd_update(
            dense_grad, 
            tf.expand_dims(indices, 1), 
            grad
        )
        return self._resource_apply_dense(dense_grad, var, apply_state)
    
    def get_config(self):
        """获取优化器配置"""
        config = super(GeneticOptimizer, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "population_size": self._serialize_hyperparameter("population_size"),
            "mutation_rate": self._serialize_hyperparameter("mutation_rate"),
            "crossover_rate": self._serialize_hyperparameter("crossover_rate"),
            "selection_pressure": self._serialize_hyperparameter("selection_pressure"),
        })
        return config
