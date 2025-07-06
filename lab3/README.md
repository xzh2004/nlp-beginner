# Lab3: 基于注意力机制的文本匹配

本项目实现了基于ESIM (Enhanced LSTM for Natural Language Inference) 的文本匹配模型，用于判断两个句子之间的关系：蕴含(entailment)、矛盾(contradiction)或中性(neutral)。

## 项目结构

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
