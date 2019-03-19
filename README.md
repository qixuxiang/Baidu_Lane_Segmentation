# Baidu_Lane_Segmentation
4th place solution in Baidu Autonomous Driving Lane Segmentation Competition

### 赛题介绍

比赛链接：[无人车车道线检测挑战赛](http://aistudio.baidu.com/aistudio/#/competition/detail/5)

数据下载地址：

- [初赛训练集](http://aistudio.baidu.com/aistudio/#/datasetDetail/1919)
- [初赛测试集](http://aistudio.baidu.com/aistudio/#/datasetDetail/2492)
- [复赛训练集](http://aistudio.baidu.com/aistudio/#/datasetDetail/3624)
- [复赛测试集](http://aistudio.baidu.com/aistudio/#/datasetdetail/3625)

### 软硬件环境

- 1080ti
- RAM>=16G
- paddlepaddle-gpu>=1.2
- CUDA>=8.0

### 复现模型

- [U-Net](https://arxiv.org/pdf/1505.04597.pdf)
- [Deeplab v3+](https://arxiv.org/pdf/1802.02611.pdf)
- [MultiResUNet](https://arxiv.org/pdf/1902.04049.pdf)

其中代码复现基本都是从Keras代码翻译到Paddle代码，所以我也贴一下上面几个模型参考的Keras源码地址：

- [Keras U-net](https://github.com/zhixuhao/unet)
- [Keras Deeplab v3+](https://github.com/mjDelta/deeplabv3plus-keras)
- [Keras MultiResUNet](https://github.com/komo135/MultiResUNet)



### 方法整理

- 正负数据不均衡：[focal loss](https://arxiv.org/pdf/1708.02002.pdf)
- 透视变换训练俯视图

### 感谢

感谢队友[universea](https://github.com/universea)提供的算力资源和强力输出，比赛之后认识了一堆大佬，这场比赛收获颇丰！
我们的比赛队伍id是Litchll，最终复赛得分0.60763，排名第四。

