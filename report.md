# NLP-Beginner 实验报告

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

| 特征类型 | 损失函数 | 学习率 | 批量大小 | 正则化 | Accuracy | Precision | Recall | F1-Score |
|----------|----------|--------|----------|--------|----------|:---------:|--------|----------|
| Bag-of-Words | Cross-Entropy | 0.01 | 16 | L2(λ=0.001) | 0.4194 | 0.4287 | 0.4194 | 0.3942 |
| N-gram | Cross-Entropy | 0.01 | 16 | L2(λ=0.001) | 0.4140 | 0.4287 | 0.4140 | 0.3849 |
| TF-IDF | Cross-Entropy | 0.01 | 16 | L2(λ=0.001) | 0.4133 | 0.4345 | 0.4133 | 0.3898 |

##### 损失函数对比实验

| 特征类型 | 损失函数 | 学习率 | 批量大小 | 正则化 | Accuracy | Precision | Recall | F1-Score |
|----------|----------|--------|----------|--------|----------|-----------|--------|----------|
| TF-IDF | Sigmoid | 0.01 | 16 | L2(λ=0.001) | 0.4086 | 0.4344 | 0.4086 | 0.3611 |
| TF-IDF | ReLU | 0.01 | 16 | L2(λ=0.001) | 0.3463 | 0.4434 | 0.3463 | 0.3797 |
| TF-IDF | Hinge | 0.01 | 16 | L2(λ=0.001) | 0.4120 | 0.4315 | 0.4120 | 0.3586 |
| TF-IDF | Cross-Entropy | 0.01 | 16 | L2(λ=0.001) | 0.4133 | 0.4345 | 0.4133 | 0.3898 |

##### 学习率对比实验

| 特征类型 | 损失函数 | 学习率 | 批量大小 | 正则化 | Accuracy | Precision | Recall | F1-Score |
|----------|----------|--------|----------|--------|----------|-----------|--------|----------|
| TF-IDF | Cross-Entropy | 0.1 | 16 | L2(λ=0.001) | 0.4129 | 0.4342 | 0.4129 | 0.3900 |
| TF-IDF | Cross-Entropy | 0.01 | 16 | L2(λ=0.001) | 0.4133 | 0.4345 | 0.4133 | 0.3898 |
| TF-IDF | Cross-Entropy | 0.001 | 16 | L2(λ=0.001) | 0.4127 | 0.4330 | 0.4079 | 0.3865 |



##### ![training_curves_softmax_lr0p1_regl2_lambda0p001_20250706_224545](./training_curves_softmax_lr0p1_regl2_lambda0p001_20250706_224545.png)

![training_curves_softmax_lr0p01_regl2_lambda0p001_20250706_082945](./training_curves_softmax_lr0p01_regl2_lambda0p001_20250706_082945.png)

![training_curves_softmax_lr0p001_regl2_lambda0p001_20250706_223800](./training_curves_softmax_lr0p001_regl2_lambda0p001_20250706_223800.png)

##### 批量大小对比实验

| 特征类型 | 损失函数 | 学习率 | 批量大小 | 正则化 | Accuracy | Precision | Recall | F1-Score |
|----------|----------|--------|----------|--------|----------|-----------|--------|----------|
| TF-IDF | Cross-Entropy | 0.01 | 1 (SGD) | L2(λ=0.001) | 0.4120 | 0.4326 | 0.4120 | 0.3891 |
| TF-IDF | Cross-Entropy | 0.01 | 16 | L2(λ=0.001) | 0.4133 | 0.4345 | 0.4133 | 0.3898 |
| TF-IDF | Cross-Entropy | 0.01 | 32 | L2(λ=0.001) | 0.4128 | 0.4336 | 0.4128 | 0.3889 |
| TF-IDF | Cross-Entropy | 0.01 | 64 | L2(λ=0.001) | 0.694 | 0.700 | 0.694 | 0.697 |

#### 实验讨论

- 理论上，TF-IDF特征优于BoW和N-gram。但实际却是BoW>N-gram>TF-IDF。推测是因为本地内存限制，导致TF-IDF提取的最大特征数量被控制在较低水平（5000），不能充分捕捉文本特征，从而导致效果不如BoW以及N-gram。
- 在损失函数对比中，发现Cross-Entropy>Hinge>Sigmoid>Relu。
- 学习率越大，曲线振幅越大；学习率越小，曲线振幅越小。因为训练采用了早停机制，发现学习率越小，曲线收敛越慢；反之越快。
- 在特征类型为TF-IDF，损失函数为Cross-Entropy，学习率为0.01的情况下，批量大小32在训练效率和性能间取得最优权衡。

## 任务二：基于深度学习的文本分类

### 实现概述
使用PyTorch实现了CNN和RNN两种深度学习文本分类模型，支持三种不同的word embedding初始化方式。

### 训练数据集

使用[Classify the sentiment of sentences from the Rotten Tomatoes dataset](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)

### 训练方法

```bash
python train.py --model cnn --embedding glove --glove_dim 300
```

### 模型架构

#### CNN模型
- **架构**：Embedding层 → 多尺度卷积核(3,4,5) → 最大池化 → Dropout → 全连接分类层
- **特点**：捕获局部特征，适合短文本分类

#### RNN模型
- **架构**：Embedding层 → LSTM层 → 注意力机制 → Dropout → 全连接分类层
- **特点**：捕获序列依赖，支持双向处理

### Word Embedding策略
1. **随机初始化**：从均匀分布初始化
2. **GloVe预训练**：使用Stanford GloVe 6B词向量(50d, 100d, 300d)
3. **Word2Vec风格**：基于语料库训练的词向量

### 实验结果
| 模型 | Embedding方式 | 验证准确率 | 测试准确率 |
|------|---------------|------------|------------|
| CNN | random | 66.44% | 66.84% |
| CNN | word2vec | 65.86% | 65.83% |
| CNN | GloVe 50d | 66.70% | 66.95% |
| CNN | GloVe 100d | 66.37% | 66.61% |
| CNN | GloVe 300d | 65.39% | 65.87% |
| RNN | random | 65.03% | 65.63% |
| RNN | GloVe 100d | 66.98% | 66.55% |
| RNN | GloVe 300d | 65.39% | 65.87% |

### 实验讨论
- 预训练词向量能够略微提升准确率。
- 实验发现GloVe Embedding dim 50 > 100 > 300，与常理相悖，推测是随着embedding维度上升，模型过拟合导致的，应当针对不同的维度采取不同的dropout值。

**参考论文**：[Convolutional Neural Networks for Sentence Classification (Kim, 2014)](https://arxiv.org/abs/1408.5882)

## 任务三：基于注意力机制的文本匹配

本项目实现了基于ESIM (Enhanced LSTM for Natural Language Inference) 的文本匹配模型，用于判断两个句子之间的关系：蕴含(entailment)、矛盾(contradiction)或中性(neutral)。

### 项目结构

```
lab3/
├── config.py              # 配置文件
├── model.py               # ESIM模型实现
├── utils.py               # 工具函数和数据预处理
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── README.md              # 项目说明
└── snli_1.0/              # SNLI数据集
    ├── snli_1.0_train.jsonl
    ├── snli_1.0_dev.jsonl
    └── snli_1.0_test.jsonl
```

### 模型架构
ESIM模型主要包含四个核心模块：

1. **输入编码层**：词嵌入 + 双向LSTM
2. **注意力交互层**：Token-to-token注意力机制
3. **推理层**：双向LSTM处理组合特征
4. **输出层**：多层感知机分类

### 训练数据集
使用[Stanford Natural Language Inference (SNLI) 数据集](https://nlp.stanford.edu/projects/snli/)：
### 实验结果
- **整体准确率(macro accuracy)**：72.65%
- **分类性能**：
  - Entailment：F1=75.68%
  - Neutral：F1=68.01%
  - Contradiction：F1=74.73%

**参考论文**： [Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038v3.pdf)

## 任务四：基于LSTM+CRF的序列标注

### 实现概述
实现了基于BiLSTM+CRF的命名实体识别(NER)模型。

### 数据集

使用[CONLL 2003数据集](https://www.clips.uantwerpen.be/conll2003/ner/)

p.s. 原版数据集处理稍显复杂，故采用huggingface上的conll 2003数据集作为替代

### 模型架构

#### BiLSTM-CRF模型
- **词嵌入层**：将词转换为向量表示（支持GloVE预训练词向量）
- **双向LSTM**：捕获上下文信息，支持多层结构
- **CRF层**：建模标签间的依赖关系，学习转移概率
- **Dropout**：防止过拟合，提高泛化能力

#### CRF实现细节
- **转移矩阵**：学习标签间的转移概率，包含START和END标签
- **前向算法**：计算所有可能路径的分数，用于训练时的损失计算
- **Viterbi算法**：找到最优标签序列，用于推理时的标签预测
- **标签约束**：正确处理PAD、START、END等特殊标签

### 实验结果

- **F1分数**：0.7444
- **Precision**: 0.7646
- **Recall**: 0.7268

### 参考论文
- [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/pdf/1603.01354)
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360)

## 任务五：基于神经网络的语言模型

实现了基于LSTM和GRU的字符级语言模型，用于生成唐诗风格的文本。

### 项目结构

```
lab5/
├── data_loader.py      # 数据加载和预处理
├── model.py           # LSTM和GRU模型定义
├── trainer.py         # 训练器
├── text_generator.py  # 文本生成器
├── train.py          # 主训练脚本
├── evaluate.py       # 模型评估脚本
├── generate.py       # 文本生成脚本
└── README.md        # 项目说明
```

### 模型架构

#### LSTM语言模型
- **字符嵌入层**：字符级向量表示
- **多层LSTM**：捕获长期依赖
- **Dropout层**：防止过拟合
- **输出层**：字符概率分布

#### GRU语言模型
- **字符嵌入层**：字符级向量表示
- **多层GRU**：简化版循环网络
- **Dropout层**：防止过拟合
- **输出层**：字符概率分布

### 训练策略
- **损失函数**：交叉熵损失
- **优化器**：Adam优化器
- **学习率调度**：动态调整学习率
- **梯度裁剪**：防止梯度爆炸

### 文本生成
支持多种生成方式：
- **自由生成**：随机生成唐诗风格文本
- **提示生成**：根据给定提示生成文本（效果欠佳）
- **诗歌生成**：生成特定格式的诗歌（默认七言绝句）

### 实验结果

测试集困惑度: 484.27
测试集损失: 6.1826

### 生成示例
```
圃不，
林人有之上。
高知何有山，
水卧跨沙载。
重知为未堂，
何君青活里。
```

### 实验讨论

实验结果表明，当前模型的诗歌生成效果未能达到预期水平。经过深入分析，我认为主要存在以下三个关键问题：

**1. 数据集规模限制**
训练数据集规模相对较小，无法为深度学习模型提供充足的训练样本，导致模型难以充分学习诗歌的语言模式和韵律特征。

**2. 模型架构复杂度不足**
当前采用的LSTM模型架构相对简单，参数量有限，缺乏足够的表达能力来捕捉诗歌创作中的复杂语义关系和韵律规律。

**3. 数据格式多样性导致的训练困难**
训练数据集中包含多种诗歌体裁（如绝句、律诗等），不同体裁的句子长度和标点符号使用规则存在显著差异。这种格式多样性使得神经网络难以准确掌握标点符号的分布规律，进而影响生成诗歌的格式规范性。

**改进建议**
针对上述问题，我建议采用**分体裁训练**的策略，即针对不同诗歌体裁（如绝句、律诗）分别训练专门的模型。这种方法能够确保每个模型专注于学习特定体裁的格式特征，从而提高生成诗歌的格式准确性和整体质量。

### 参考资料：

 https://github.com/YC-Coder-Chen/Tang-Poetry-Generator/tree/master