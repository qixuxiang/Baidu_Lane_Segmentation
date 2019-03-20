# Baidu_Lane_Segmentation
4th place solution in Baidu Autonomous Driving Lane Segmentation Competition

## 赛题介绍

比赛链接：[无人车车道线检测挑战赛](http://aistudio.baidu.com/aistudio/#/competition/detail/5)

数据下载地址：

- [初赛训练集](http://aistudio.baidu.com/aistudio/#/datasetDetail/1919)
- [初赛测试集](http://aistudio.baidu.com/aistudio/#/datasetDetail/2492)
- [复赛训练集](http://aistudio.baidu.com/aistudio/#/datasetDetail/3624)
- [复赛测试集](http://aistudio.baidu.com/aistudio/#/datasetdetail/3625)

## 软硬件环境

- 1080ti
- RAM>=16G
- paddlepaddle-gpu>=1.2
- CUDA>=8.0

## 复现模型

- [U-Net](https://arxiv.org/pdf/1505.04597.pdf)
- [Deeplab v3+](https://arxiv.org/pdf/1802.02611.pdf)
- [MultiResUNet](https://arxiv.org/pdf/1902.04049.pdf)
- [DenseUNet](https://arxiv.org/pdf/1608.06993.pdf)
- [Pyramid-Attention-Networks](https://arxiv.org/pdf/1805.10180.pdf)

其中代码复现基本都是从Keras代码翻译到Paddle代码，所以我也贴一下上面几个模型参考的Keras源码地址：

- [Keras U-net](https://github.com/zhixuhao/unet)
- [Keras Deeplab v3+](https://github.com/mjDelta/deeplabv3plus-keras)
- [Keras MultiResUNet](https://github.com/komo135/MultiResUNet)
- [DenseUNet](https://github.com/DeepTrial/Retina-VesselNet)
- [Pyramid-Attention-Networks](https://www.jianshu.com/p/c5eb9976866f)

## 运行

### 准备

#### 数据

请下载上面数据链接中的所有数据，把训练数据zip包放到`data/ApolloDatas/train`目录下面并解压，把测试数据zip包放到`data/ApolloDatas/test`目录下面并解压。

#### 预训练模型

请戳[链接](https://pan.baidu.com/share/init?surl=7wgDUGFLDw7lQkr0M-Ob6g)下载params文件放到`params`文件夹下，提取码`4qsa `。

### 训练

```
python train.py --model=unet 
```

这里训练默认为从头训练，其中unet可以替换为`deeplab_v3p`，`deeplabv3p_ours`，`multires_unet`，和`pannet`，分别对应上面几个模型。

如果用预训练模型请在`train.py`下把`pretrain_model`的值改为1。

### 测试

```
python predict.py.py --model=unet 
```

其中unet可以替换为`deeplab_v3p`，`deeplabv3p_ours`，`multires_unet`，和`pannet`，分别对应上面几个模型。

## 方法整理

- 正负数据不均衡：[focal loss](https://arxiv.org/pdf/1708.02002.pdf)
- 透视变换训练俯视图
- SCSE: [Concurrent Spatial and Channel ‘Squeeze &Excitation’ in Fully Convolutional Networks](https://arxiv.org/pdf/1803.02579.pdf)

## 感谢

感谢队友[universea](https://github.com/universea)提供的算力资源和强力输出，比赛之后认识了一堆大佬，这场比赛收获颇丰！

我们的比赛队伍id是Litchll，最终复赛得分0.60763，排名第四。

## 其他问题

因为模型较大，所以运行时如果内存不足可能报错，这时可尝试调节减小输入的size让代码运行。

欢迎任何形式的PR和issue。

