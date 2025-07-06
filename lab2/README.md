- ## 任务二：基于深度学习的文本分类

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
  | ---- | ------------- | ---------- | ---------- |
  | CNN  | random        | 66.44%     | 66.84%     |
  | CNN  | word2vec      | 65.86%     | 65.83%     |
  | CNN  | GloVe 50d     | 66.70%     | 66.95%     |
  | CNN  | GloVe 100d    | 66.37%     | 66.61%     |
  | CNN  | GloVe 300d    | 65.39%     | 65.87%     |
  | RNN  | random        | 65.03%     | 65.63%     |
  | RNN  | GloVe 100d    | 66.98%     | 66.55%     |
  | RNN  | GloVe 300d    | 65.39%     | 65.87%     |

  ### 实验讨论

  - 预训练词向量能够略微提升准确率。
  - 实验发现GloVe Embedding dim 50 > 100 > 300，与常理相悖，推测是随着embedding维度上升，模型过拟合导致的，应当针对不同的维度采取不同的dropout值。

  **参考论文**：[Convolutional Neural Networks for Sentence Classification (Kim, 2014)](https://arxiv.org/abs/1408.5882)