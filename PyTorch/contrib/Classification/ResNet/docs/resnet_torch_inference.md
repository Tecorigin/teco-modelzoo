# ResNet

## 1. 模型概述

ResNet是一种深度卷积神经网络模型，采用了残差网络（ResNet）的结构，通过引入残差块（Residual Block）以解决深度神经网络训练中的梯度消失和表示瓶颈问题。ResNet模型在各种计算机视觉任务中表现优异，如图像分类、目标检测和语义分割等。由于其良好的性能和广泛的应用，ResNet已成为深度学习和计算机视觉领域的重要基础模型之一。

当前，TecoPytorch支持推理的ResNet模型包括：ResNet50。

## 2. 快速开始

使用本模型执行模型推理的主要流程如下：
1. 基础环境安装：介绍推理前需要完成的基础环境检查和安装。
2. 安装第三方依赖：介绍如何安装模型推理所需的第三方依赖。
3. 获取模型文件：介绍如何获取推理所需的模型文件。
4. 获取数据集：介绍如何获取推理所需的数据集。
5. 数据集精度验证：介绍如何验证推理精度。

### 2.1 基础环境安装

请参考推理首页的[基础环境安装](../README.md)章节，完成推理前的基础环境检查和安装。

### 2.2 安装第三方依赖

1. 执行以下命令，进入第三方依赖安装脚本所在目录。

   ```shell
   cd <modelzoo_dir>/PyTorch/Classification/ResNet
   ```

2. 执行以下命令，安装第三方依赖。

   ```shell
   pip install -r requirements.txt
   ```

   **注意**：若速度过慢，可加上`-i`参数指定源。


### 2.3 获取模型文件

1. 根据要推理的模型，点击下载模型权重。

   - [resnet50](https://pan.baidu.com/s/1ru4IOsNulb_ukH7cFJszQg?pwd=hqxd)


### 2.4 获取数据集

您可以通过以下方式获取推理所需的数据集：
- 使用[ImageNet数据集](https://image-net.org/download-images)，用于模型推理和推理精度验证。

### 2.5 数据集精度验证

请提前准备好ImageNet数据集，执行以下命令，获得推理精度数据。
1. 执行以下命令，进入模型评估目录。

   ```shell
   cd <modelzoo_dir>/PyTorch/Classification/ResNet/tools
   ```

2.单核组推理：

```bash
python eval_dist.py --target sdaa --data-path /path_to/imagenet --half --batch-size 128
```

3.分布式推理：

```bash
torchrun --nproc_per_node=4 eval_dist.py --target sdaa --data-path /path_to/imagenet --half --batch-size 128
```
4.权重int8推理：
```bash
torchrun --nproc_per_node=4 eval_dist.py  --weight-int8 --target sdaa --data-path /path_to/imagenet --half --batch-size 128
```

部分精度结果如下：

```shell
Final Evaluation Accuracy: 78.5308
Total Infer Time: 9.3447s
Avg Infer Time: 0.0954s
Total System Infer SPS: 5348.8017
```

 结果说明：

| 参数 | 说明 |
| ------------- | ------------- |
| Final Evaluation Accuracy   | 数据集验证精度  |
| Total Infer Time | 总推理计算时间(s) |
| Avg Infer Time | 平均推理计算时间(s)  |
| Total System Infer SPS | 吞吐量(samples/s) |

