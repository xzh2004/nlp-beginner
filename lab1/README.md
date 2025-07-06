## 任务一：基于机器学习的文本分类

### 概述

实现了基于Logistic Regression和Softmax Regression的文本分类系统，使用NumPy进行底层实现，支持多种特征提取方法（Bag-of-Words (BoW)、N-gram、TF-IDF）和损失函数（Sigmoid、ReLU、Hinge、Cross-Entropy）。

### 训练数据集

使用[Classify the sentiment of sentences from the Rotten Tomatoes dataset](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)

### 训练方法

方法1: 配置文件

```bash
python train.py --config high_accuracy_config.json
```

方法2: 命令行参数

```bash
python train.py --loss_function hinge --learning_rate 0.01
```

### 实验设计

1. **特征实验**：比较BoW、N-gram、TF-IDF和组合特征的效果
2. **学习率实验**：测试0.001、0.01以及0.1不同学习率对收敛的影响
3. **批量大小实验**：比较SGD(1)、16、32、64、128批量大小的效果

### 实验结果

##### 特征提取方法对比实验

| 特征类型     | 损失函数      | 学习率 | 批量大小 | 正则化      | Accuracy | Precision | Recall | F1-Score |
| ------------ | ------------- | ------ | -------- | ----------- | -------- | :-------: | ------ | -------- |
| Bag-of-Words | Cross-Entropy | 0.01   | 16       | L2(λ=0.001) | 0.4194   |  0.4287   | 0.4194 | 0.3942   |
| N-gram       | Cross-Entropy | 0.01   | 16       | L2(λ=0.001) | 0.4140   |  0.4287   | 0.4140 | 0.3849   |
| TF-IDF       | Cross-Entropy | 0.01   | 16       | L2(λ=0.001) | 0.4133   |  0.4345   | 0.4133 | 0.3898   |

##### 损失函数对比实验

| 特征类型 | 损失函数      | 学习率 | 批量大小 | 正则化      | Accuracy | Precision | Recall | F1-Score |
| -------- | ------------- | ------ | -------- | ----------- | -------- | --------- | ------ | -------- |
| TF-IDF   | Sigmoid       | 0.01   | 16       | L2(λ=0.001) | 0.4086   | 0.4344    | 0.4086 | 0.3611   |
| TF-IDF   | ReLU          | 0.01   | 16       | L2(λ=0.001) | 0.3463   | 0.4434    | 0.3463 | 0.3797   |
| TF-IDF   | Hinge         | 0.01   | 16       | L2(λ=0.001) | 0.4120   | 0.4315    | 0.4120 | 0.3586   |
| TF-IDF   | Cross-Entropy | 0.01   | 16       | L2(λ=0.001) | 0.4133   | 0.4345    | 0.4133 | 0.3898   |

##### 学习率对比实验

| 特征类型 | 损失函数      | 学习率 | 批量大小 | 正则化      | Accuracy | Precision | Recall | F1-Score |
| -------- | ------------- | ------ | -------- | ----------- | -------- | --------- | ------ | -------- |
| TF-IDF   | Cross-Entropy | 0.1    | 16       | L2(λ=0.001) | 0.4129   | 0.4342    | 0.4129 | 0.3900   |
| TF-IDF   | Cross-Entropy | 0.01   | 16       | L2(λ=0.001) | 0.4133   | 0.4345    | 0.4133 | 0.3898   |
| TF-IDF   | Cross-Entropy | 0.001  | 16       | L2(λ=0.001) | 0.4127   | 0.4330    | 0.4079 | 0.3865   |



##### ![training_curves_softmax_lr0p1_regl2_lambda0p001_20250706_224545](./training_curves_softmax_lr0p1_regl2_lambda0p001_20250706_224545.png)

![training_curves_softmax_lr0p01_regl2_lambda0p001_20250706_082945](./training_curves_softmax_lr0p01_regl2_lambda0p001_20250706_082945.png)

![training_curves_softmax_lr0p001_regl2_lambda0p001_20250706_223800](./training_curves_softmax_lr0p001_regl2_lambda0p001_20250706_223800.png)

##### 批量大小对比实验

| 特征类型 | 损失函数      | 学习率 | 批量大小 | 正则化      | Accuracy | Precision | Recall | F1-Score |
| -------- | ------------- | ------ | -------- | ----------- | -------- | --------- | ------ | -------- |
| TF-IDF   | Cross-Entropy | 0.01   | 1 (SGD)  | L2(λ=0.001) | 0.4120   | 0.4326    | 0.4120 | 0.3891   |
| TF-IDF   | Cross-Entropy | 0.01   | 16       | L2(λ=0.001) | 0.4133   | 0.4345    | 0.4133 | 0.3898   |
| TF-IDF   | Cross-Entropy | 0.01   | 32       | L2(λ=0.001) | 0.4128   | 0.4336    | 0.4128 | 0.3889   |
| TF-IDF   | Cross-Entropy | 0.01   | 64       | L2(λ=0.001) | 0.694    | 0.700     | 0.694  | 0.697    |

#### 实验讨论

- 理论上，TF-IDF特征优于BoW和N-gram。但实际却是BoW>N-gram>TF-IDF。推测是因为本地内存限制，导致TF-IDF提取的最大特征数量被控制在较低水平（5000），不能充分捕捉文本特征，从而导致效果不如BoW以及N-gram。
- 在损失函数对比中，发现Cross-Entropy>Hinge>Sigmoid>Relu。
- 学习率越大，曲线振幅越大；学习率越小，曲线振幅越小。因为训练采用了早停机制，发现学习率越小，曲线收敛越慢；反之越快。
- 在特征类型为TF-IDF，损失函数为Cross-Entropy，学习率为0.01的情况下，批量大小32在训练效率和性能间取得最优权衡。