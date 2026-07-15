# ResNet

## 1. 模型概述
ResNet是一种深度卷积神经网络模型，采用了残差网络（ResNet）的结构，通过引入残差块（Residual Block）以解决深度神经网络训练中的梯度消失和表示瓶颈问题。ResNet模型在各种计算机视觉任务中表现优异，如图像分类、目标检测和语义分割等。由于其良好的性能和广泛的应用，ResNet已成为深度学习和计算机视觉领域的重要基础模型之一。
当前支持的ResNet模型包括：ResNet50、ResNet18 和ResNet101。

- 参考实现：
    ```
    url=https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/GPUNet
    commit_id=1dbad27d1c53a8f2bb13c0d4124a853f3df1f30b
    ```

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集

ResNet50运行在ImageNet 1k上，这是一个来自ILSVRC挑战赛的广受欢迎的图像分类数据集。您可以点击[此链接](https://image-net.org/download-images)从公开网站中下载数据集。

#### 2.2.2 解压数据集

- 执行以下命令，解压训练数据集。
    ```
    mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
    tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
    find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
    cd ..
    ```
- 执行以下命令，解压测试数据并将图像移动到子文件夹中。
    ```
    mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
    wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
    ```

解压后的数据集目录结构如下所示：
```
├── ImageNet2012
        ├──train
            ├──类别1
                │──图片1
                │──图片2
                │   ...
            ├──类别2
                │──图片1
                │──图片2
                │   ...
            ├──...
        ├──val
            ├──类别1
                │──图片1
                │──图片2
                │   ...
            ├──类别2
                │──图片1
                │──图片2
                │   ...
```


### 2.3 构建环境

所使用的环境下已经包含 PyTorch 框架虚拟环境。

1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch27_env_py310
    
    #龙芯国产环境无conda
    ```

2. 执行以下命令，安装python依赖。
    ```
    pip install -r requirements.txt
    ```

### 2.4 启动训练
1. 进入训练脚本所在目录。
    ```
    cd <Teco-ModelZoo_path>/PyTorch/contrib/Classification/ResNet/run_scripts
    ```

2. 运行训练。该模型支持单机单卡、单机四卡、单机八卡、两机八卡。

   - 单机单卡
     ```
     python run_resnet.py --model_name resnet50 --nproc_per_node 4 --bs 64 --lr 0.256 --device sdaa --epoch 50 --dataset_path /datasets/imagenet/ --grad_scale True --autocast True
     ```

   - 单机四卡
       ```
       python run_resnet.py --model_name resnet50 --nproc_per_node 16 --bs 64 --lr 1.024 --device sdaa --epoch 50 --dataset_path /datasets/imagenet/ --grad_scale True --autocast True
       ```
   
    - 单机八卡
       ```
       python run_resnet.py --model_name resnet50 --nproc_per_node 32 --bs 64 --lr 2.048 --device sdaa --epoch 90 --dataset_path /datasets/imagenet/ --grad_scale True --autocast True
       ```
   
   - 两机八卡
       
     两机运行时，需要在两台机器节点上分别执行训练命令。运行时请将`ip_address`替换为主节点的IP地址，不同节点的命令只有`node_rank`不同（其中，主节点的`node_rank`为0），其余参数取值应相同。
       
       **node0：**
       ```
       python run_resnet.py --model_name resnet50 --nnode 2 --node_rank 0 --master_addr <ip_address> --master_port 29500 --nproc_per_node 16 --bs 64 --lr 2.048 --device sdaa --epoch 50 --dataset_path /datasets/imagenet/ --grad_scale True --autocast True
       ```
       **node1:**
       ```
       python run_resnet.py --model_name resnet50 --nnode 2 --node_rank 1 --master_addr <ip_address> --master_port 29500 --nproc_per_node 16 --bs 64 --lr 2.048 --device sdaa --epoch 50 --dataset_path /datasets/imagenet/ --grad_scale True --autocast True
       ```
 
    训练命令参数说明参考[README](run_scripts/README.md)。



### 2.5 训练结果

|加速卡数量  |模型 | 混合精度 |Batch size|Shape| 性能(imgs/s) | AccTop1 | sdaa训练耗时 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|1| ResNet50|是|256|224*224| 1125 imgs/s | 76.09% | 7h |

