# 2025.4.1 目前进展
## 多层膜模型
    目前三层膜系统的神经网络模型已经训练完毕，这里做一个简要总结

#####  网络架构思路
```markdown
    三层膜模型是一个比较简单的多层膜物理系统，
    由三个膜层组成，每个膜层都有自己的折射率和厚度。
    因此，我们采用了全连接神经网络。
    同时，我们不需要很深的神经网络，只用了两个隐藏层，并且这两个隐藏层神经元数量较少。
    另外，通过折射率和厚度的输入，得到反射率输出，这是一个典型的回归问题。
    因此，在初期的代码可行性验证阶段，我先是采用了huber_loss作为损失函数。
    但是，因为我们的数据量很小，初期只有1000个数据，
    因此需要物理信息神经网络来提高模型的泛化能力。
    所以我们后来改用了PINN，
    相比于huber_loss，PINN可以更好地利用物理知识来约束模型。
    因此，我们得到的模型在测试数据上的表现更好。
```
#####  网络结构改进
```markdown
考虑到网络中4个主要的非线性模型，其中，输入有7个神经元，1个输出神经元，
两个隐藏层，第一个隐藏层20个神经元，第二个隐藏层10个神经元。
由此可见，我们使用了较少的参数量。
另外，我们还加入了Dropout和BatchNormalization正则化技术防止过拟合。
![model_architecture](models/best_pinn_model_architecture_2.png "model_architecture")
```
#####  物理知识训练
```markdown
该模型训练的主要数据来源于FDTD模拟。
我们用了1000个数据进行了训练。
我们用K折交叉验证来评估数据集分为5份，交叉训练，然后取最好模型。
由于网络参数较少，所以我们采用了Adam优化器，这是一个比较简单且有效的优化器。
虽然数据量很好地解决了大时间1000个数据，数据梯度缺乏问题，达到了较好的训练效果。
在测试中，我们发现Dropout起了不可或缺的作用。因此
模型的损失只有0.06左右。
![training_loss_curve](models/training_loss_curve_2.png "training_loss_curve")
```
## 硬件配置
    模型的训练和测试在GPU上进行：
##### GPU配置
```markdown
    GPU: NVIDIA RTX 4060
    这是一块很不错的显卡，极大地提高了训练速度。
##### 硬件选择思路
    由于神经网络的计算量很大，并且其中有很多并行计算，
    而GPU在并行计算方面有得天独厚的优势，
    因此选择RTX 4060作为训练设备。
    
##### 硬件配置过程
```markdown
    硬件的配置是充满曲折的。
    由于电脑的系统是Windows，
    而高版本TensorFlow和Keras需要在Linux或macOS系统上运行，
    因此需要在Windows系统上安装虚拟机，
    由于这种方法太过麻烦，所以我们选择了较低版本的TensorFlow和Keras，
    这样就可以在Windows系统上直接运行了。
    但后来又产出了CUDA Toolkit的兼容性问题，
    经过很久的尝试，最终我通过复制dll文件这种土办法完成了配置。
    这是一个很麻烦的配置过程，也是选择tensorflow为数不多的缺点之一。
    ![cuda_compatibility](record_md/cuda_compatibility.png "cuda_compatibility")
