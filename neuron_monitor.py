import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

class NeuronWeightMonitor(tf.keras.callbacks.Callback):
    """监测每个神经元权重变化的回调类
    
    追踪每个神经元的权重变化，并生成详细的可视化和数据分析
    """
    def __init__(self, 
                output_dir='neuron_weights', 
                freq=1,
                max_neurons_per_layer=10,
                layer_name_filter='dense'):
        """初始化神经元监测器
        
        Args:
            output_dir: 输出目录
            freq: 记录频率（每多少个epoch记录一次）
            max_neurons_per_layer: 每层最多监测的神经元数量
            layer_name_filter: 层名称过滤器，只监测包含此字符串的层
        """
        super().__init__()
        self.output_dir = output_dir
        self.freq = freq
        self.max_neurons_per_layer = max_neurons_per_layer
        self.layer_name_filter = layer_name_filter
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 存储神经元权重历史的字典
        # 格式: {layer_id: {neuron_id: {epoch: weight_vector}}}
        self.neuron_weights = defaultdict(lambda: defaultdict(dict))
        
        # 存储神经元统计信息
        # 格式: {layer_id: {neuron_id: {epoch: {'mean': mean, 'std': std, 'min': min, 'max': max}}}}
        self.neuron_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        
        print(f"已初始化神经元权重监测器 - 将保存到 {output_dir}")
        
    def on_train_begin(self, logs=None):
        """训练开始时的初始化"""
        # 扫描模型层和神经元
        self._scan_model_neurons()
        
    def _scan_model_neurons(self):
        """扫描模型的层和神经元，选择要监测的神经元"""
        self.monitored_neurons = {}
        
        for i, layer in enumerate(self.model.layers):
            # 只关注指定类型的层
            if not layer.weights or self.layer_name_filter not in layer.name.lower():
                continue
                
            layer_id = f"{i}_{layer.name}"
            self.monitored_neurons[layer_id] = []
            
            # 对于每个权重矩阵 (通常是kernel和bias)
            for j, weight in enumerate(layer.weights):
                weight_name = weight.name.split('/')[-1].replace(':', '_')
                
                if 'kernel' in weight_name:  # 只追踪连接权重，不追踪偏置
                    shape = weight.shape.as_list()
                    
                    if len(shape) == 2:  # Dense层的权重矩阵
                        input_dim, output_dim = shape
                        
                        # 确定要监测的神经元数量
                        neurons_to_monitor = min(output_dim, self.max_neurons_per_layer)
                        
                        # 均匀选择神经元以确保覆盖范围
                        if neurons_to_monitor < output_dim:
                            neuron_indices = np.linspace(0, output_dim-1, neurons_to_monitor, dtype=int)
                        else:
                            neuron_indices = range(output_dim)
                            
                        for neuron_idx in neuron_indices:
                            neuron_id = f"{weight_name}_neuron_{neuron_idx}"
                            self.monitored_neurons[layer_id].append({
                                'weight_name': weight_name,
                                'neuron_idx': neuron_idx,
                                'neuron_id': neuron_id,
                                'input_dim': input_dim
                            })
        
        # 打印监测信息                
        total_neurons = sum(len(neurons) for neurons in self.monitored_neurons.values())
        print(f"神经元监测器将追踪 {len(self.monitored_neurons)} 层中的 {total_neurons} 个神经元")
    
    def on_epoch_end(self, epoch, logs=None):
        """每个epoch结束时记录神经元权重"""
        if epoch % self.freq != 0:
            return
            
        # 收集所有监测神经元的当前权重
        for layer_id, neurons in self.monitored_neurons.items():
            layer_idx = int(layer_id.split('_')[0])
            layer = self.model.layers[layer_idx]
            
            for neuron_info in neurons:
                weight_name = neuron_info['weight_name']
                neuron_idx = neuron_info['neuron_idx']
                neuron_id = neuron_info['neuron_id']
                
                # 获取权重矩阵
                for weight in layer.weights:
                    if weight_name in weight.name:
                        weight_matrix = weight.numpy()
                        
                        # 提取这个神经元的所有输入权重
                        if len(weight_matrix.shape) == 2:  # Dense层
                            # shape: [input_dim, output_dim]
                            neuron_weights = weight_matrix[:, neuron_idx]
                            
                            # 计算统计信息
                            mean_val = np.mean(neuron_weights)
                            std_val = np.std(neuron_weights)
                            min_val = np.min(neuron_weights)
                            max_val = np.max(neuron_weights)
                            
                            # 保存权重向量
                            self.neuron_weights[layer_id][neuron_id][epoch] = neuron_weights
                            
                            # 保存统计信息
                            self.neuron_stats[layer_id][neuron_id][epoch] = {
                                'mean': mean_val,
                                'std': std_val,
                                'min': min_val,
                                'max': max_val
                            }
        
        # 每5个epoch生成一次可视化
        if epoch % 5 == 0 or epoch == 0:
            self._generate_neuron_visualizations(epoch)
            self._save_neuron_stats(epoch)
            print(f"Epoch {epoch}: 已生成 {len(self.monitored_neurons)} 层神经元权重的可视化")
    
    def _generate_neuron_visualizations(self, epoch):
        """生成神经元权重可视化"""
        # 为每一层生成一个图
        for layer_id, neurons in self.monitored_neurons.items():
            if not neurons:
                continue
                
            # 创建图形
            plt.figure(figsize=(15, 10))
            plt.suptitle(f"层 {layer_id} 中的神经元权重 - Epoch {epoch}", fontsize=16)
            
            # 计算子图布局
            n_neurons = len(neurons)
            n_cols = min(3, n_neurons)
            n_rows = (n_neurons + n_cols - 1) // n_cols
            
            # 绘制每个神经元的权重直方图
            for i, neuron_info in enumerate(neurons):
                neuron_id = neuron_info['neuron_id']
                
                # 确保有数据
                if epoch not in self.neuron_weights[layer_id][neuron_id]:
                    continue
                    
                weights = self.neuron_weights[layer_id][neuron_id][epoch]
                stats = self.neuron_stats[layer_id][neuron_id][epoch]
                
                ax = plt.subplot(n_rows, n_cols, i+1)
                
                # 绘制直方图
                plt.hist(weights, bins=30, alpha=0.7)
                plt.axvline(stats['mean'], color='red', linestyle='dashed', linewidth=1)
                
                # 添加统计信息
                plt.title(f"神经元 {neuron_info['neuron_idx']}")
                stats_text = f"均值: {stats['mean']:.4f}\n标准差: {stats['std']:.4f}\n范围: [{stats['min']:.4f}, {stats['max']:.4f}]"
                plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                            va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # 保存图像
            fig_path = os.path.join(self.output_dir, f"epoch_{epoch:03d}_layer_{layer_id}_neurons.png")
            plt.savefig(fig_path, dpi=150)
            plt.close()
            
            # 如果有足够的历史数据，生成时间序列图
            if epoch >= 5:
                self._generate_time_series_plots(layer_id, epoch)
    
    def _generate_time_series_plots(self, layer_id, current_epoch):
        """生成神经元权重随时间变化的图表"""
        neurons = self.monitored_neurons[layer_id]
        
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"层 {layer_id} 中神经元权重统计随时间变化 (到Epoch {current_epoch})", fontsize=16)
        
        # 计算子图布局
        n_neurons = len(neurons)
        n_cols = min(3, n_neurons)
        n_rows = (n_neurons + n_cols - 1) // n_cols
        
        for i, neuron_info in enumerate(neurons):
            neuron_id = neuron_info['neuron_id']
            neuron_stats = self.neuron_stats[layer_id][neuron_id]
            
            # 收集时间序列数据
            epochs = sorted(neuron_stats.keys())
            means = [neuron_stats[e]['mean'] for e in epochs]
            stds = [neuron_stats[e]['std'] for e in epochs]
            mins = [neuron_stats[e]['min'] for e in epochs]
            maxs = [neuron_stats[e]['max'] for e in epochs]
            
            # 绘制时间序列
            ax = plt.subplot(n_rows, n_cols, i+1)
            plt.plot(epochs, means, 'r-', label='均值')
            plt.fill_between(epochs, 
                          np.array(means) - np.array(stds),
                          np.array(means) + np.array(stds),
                          color='r', alpha=0.2)
            plt.plot(epochs, mins, 'g--', label='最小值')
            plt.plot(epochs, maxs, 'b--', label='最大值')
            
            plt.title(f"神经元 {neuron_info['neuron_idx']}")
            plt.xlabel("Epoch")
            plt.ylabel("权重值")
            plt.legend(loc='best', fontsize='small')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存图像
        fig_path = os.path.join(self.output_dir, f"epoch_{current_epoch:03d}_layer_{layer_id}_time_series.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
    
    def _save_neuron_stats(self, epoch):
        """保存神经元统计数据到CSV文件"""
        # 为当前epoch创建一个数据框
        data = []
        
        for layer_id, neurons in self.monitored_neurons.items():
            for neuron_info in neurons:
                neuron_id = neuron_info['neuron_id']
                
                # 确保有数据
                if epoch not in self.neuron_stats[layer_id][neuron_id]:
                    continue
                
                stats = self.neuron_stats[layer_id][neuron_id][epoch]
                
                data.append({
                    'epoch': epoch,
                    'layer_id': layer_id,
                    'neuron_idx': neuron_info['neuron_idx'],
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max']
                })
        
        # 创建数据框并保存
        if data:
            df = pd.DataFrame(data)
            stats_path = os.path.join(self.output_dir, f"epoch_{epoch:03d}_neuron_stats.csv")
            df.to_csv(stats_path, index=False)
            
    def on_train_end(self, logs=None):
        """训练结束时的总结报告"""
        # 生成总结报告
        summary_data = []
        
        for layer_id, neurons in self.monitored_neurons.items():
            for neuron_info in neurons:
                neuron_id = neuron_info['neuron_id']
                neuron_stats = self.neuron_stats[layer_id][neuron_id]
                
                if not neuron_stats:
                    continue
                    
                # 获取最后一个epoch的统计信息
                last_epoch = max(neuron_stats.keys())
                stats = neuron_stats[last_epoch]
                
                # 计算权重变化率
                if len(neuron_stats) > 1:
                    first_epoch = min(neuron_stats.keys())
                    first_mean = neuron_stats[first_epoch]['mean']
                    last_mean = stats['mean']
                    mean_change = (last_mean - first_mean) / max(abs(first_mean), 1e-10) * 100
                else:
                    mean_change = 0
                
                summary_data.append({
                    'layer_id': layer_id,
                    'neuron_idx': neuron_info['neuron_idx'],
                    'final_mean': stats['mean'],
                    'final_std': stats['std'],
                    'final_min': stats['min'],
                    'final_max': stats['max'],
                    'mean_change_percent': mean_change
                })
        
        # 创建汇总数据框并保存
        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_path = os.path.join(self.output_dir, "neuron_weight_summary.csv")
            df.to_csv(summary_path, index=False)
            
            print(f"神经元权重监测完成，汇总数据已保存到 {summary_path}")
            print(f"最大平均权重变化: {df['mean_change_percent'].abs().max():.2f}%")
