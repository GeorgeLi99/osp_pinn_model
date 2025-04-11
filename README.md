# 多层薄膜光学特性预测系统 (PINN-OSP)

基于物理信息神经网络(Physics-Informed Neural Network, PINN)的多层薄膜光学特性预测系统。该系统能够根据薄膜的材料折射率和厚度参数，预测其光学透射率。

![版本](https://img.shields.io/badge/版本-1.0.0-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange)
![Python](https://img.shields.io/badge/Python-3.8+-green)

## 项目介绍

多层薄膜结构在光学领域有着广泛应用，如滤光片、反射镜和防反射涂层等。传统的光学性能计算通常基于物理方程和数值方法，计算复杂且耗时。本项目利用深度学习与物理模型相结合的PINN方法，不仅能够快速预测多层薄膜的光学特性，而且保证了预测结果符合物理规律。

### 系统特点

- **物理约束优化**：将物理定律作为损失函数的一部分，确保模型结果符合光学物理规律
- **高精度预测**：利用批归一化、He初始化等深度学习技术提高预测精度
- **灵活配置**：所有模型参数和训练设置均可通过配置文件灵活调整
- **全面监控**：整合TensorBoard和自定义权重监控，实时跟踪训练过程
- **交叉验证**：采用K折交叉验证提高模型泛化能力和稳定性
- **可视化分析**：自动生成预测vs真实值对比图、分布图和权重变化图等

## 数据格式

系统接受CSV格式的输入数据，每行包含多层薄膜的参数和对应的透射率：
- 输入参数：各层材料的折射率(n)和厚度(h)
- 输出参数：透射率T(λ)

## 安装说明

### 前置要求

- Python 3.8+
- Git
- Git LFS（用于管理大型训练数据文件）

### Git LFS设置

本项目使用Git LFS（Large File Storage）管理大型数据文件，如反射率数据集。请按照以下步骤设置：

1. 安装Git LFS：

   ```bash
   # Windows
   # 通过官网下载安装: https://git-lfs.github.com/
   
   # MacOS
   brew install git-lfs
   
   # Linux
   sudo apt-get install git-lfs
   ```

2. 设置Git LFS：

   ```bash
   git lfs install
   ```

3. 克隆仓库后，大型文件将自动被Git LFS管理：

   ```bash
   git clone https://github.com/yourusername/pinn-osp.git
   ```

### 安装依赖

```bash
# 安装所需Python包
pip install -r requirements.txt
```

## 项目结构

```text
pinn-osp/
├── config.py             # 配置文件
├── pinn_model.py         # PINN模型实现
├── correct_model_clean.py # 神经网络模型（优化版本）
├── generate_training_data.py # 训练数据生成
├── data/                 # 数据目录
│   ├── train_data.csv    # 训练数据
│   └── reflectance_data.zip # 反射率数据(LFS)
├── models/               # 保存的模型
├── results/              # 结果输出
└── README.md            # 项目文档
```

## 使用方法

### 训练模型

```bash
python correct_model_clean.py
```

### 生成训练数据

```bash
python generate_training_data.py
```

### 使用PINN模型

```bash
python pinn_model.py
```

## 模型配置

模型的各项参数都集中在各个文件的顶部，可以根据需要调整：

- **网络结构参数**：层数、神经元数量、激活函数
- **训练参数**：批量大小、学习率、训练轮数
- **优化器参数**：选择Adam、SGD或自定义的遗传算法优化器

## 开发记录

- **2025-04**：实现25层深度网络，大幅提高拟合能力
- **2025-03**：添加遗传算法优化器
- **2025-02**：优化权重初始化方法
- **2025-01**：项目初始化，实现基础PINN模型

## 许可证

MIT License

## 数据示例

```csv
n1(SiO2),n2(Si3N4),n3(Al2O3),h1(SiO2)/nm,h2(Si3N4)/nm,h3(Al2O3)/nm,T()
2.12699,4.06632,3.1262,0.1,0.1,0.1,0.1405287674405466
...
```

## 核心模块

- `config.py` - 系统配置文件，包含所有可调整参数
- `process_data.py` - 数据处理模块，负责加载和预处理数据
- `loss_pinn.py` - 物理信息损失函数定义
- `pinn_model.py` - 主模型定义和训练模块
- `predict.py` - 预测和评估模块
- `data/` - 数据文件夹
- `models/` - 模型保存和输出文件夹

## 如何使用

1. **配置参数**：根据需要修改`config.py`中的参数
2. **训练模型**：运行`python pinn_model.py`进行模型训练
3. **监控训练**：使用TensorBoard查看训练过程：`tensorboard --logdir=models/logs`
4. **预测测试**：运行`python predict.py`对新数据进行预测

## 系统要求

- Python 3.6+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## 开发说明

模型架构采用多层神经网络，特别优化了以下方面：
- 使用批归一化加速训练和提高稳定性
- 在批归一化层前的Dense层中禁用偏置以减少冗余
- 使用He初始化避免梯度消失/爆炸问题
- 采用tanh作为输出层激活函数以适应数据范围
- 集成物理约束确保预测结果遵循薄膜光学定律