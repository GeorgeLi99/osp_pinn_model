"""
模型结构可视化工具
用于生成神经网络结构图
"""

import os
import tensorflow as tf

def visualize_model_structure(model, save_path=None, results_dir='results'):
    """绘制模型结构图并保存
    
    Args:
        model: 要可视化的模型
        save_path: 图片保存路径，默认为模型结构图.png
        results_dir: 结果保存目录
    """
    if save_path is None:
        save_path = os.path.join(results_dir, "model_structure.png")
        
    # 确保是.png或.jpg格式
    if not (save_path.endswith('.png') or save_path.endswith('.jpg')):
        save_path += '.png'
        
    print(f"\n生成模型结构图: {save_path}")
    
    try:
        # 使用TensorFlow的工具绘制模型图
        tf.keras.utils.plot_model(
            model,
            to_file=save_path,
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",  # 从上到下布局
            expand_nested=True,
            dpi=96,
            layer_range=None,
            show_layer_activations=True
        )
        print(f"模型结构图已保存到: {save_path}")
    except Exception as e:
        print(f"绘制模型结构图时出错: {e}")
        print("请确保已安装 pydot 和 graphviz，可使用 'pip install pydot graphviz' 安装")
