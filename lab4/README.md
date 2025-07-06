# 任务四：基于LSTM+CRF的序列标注

本项目实现了基于LSTM+CRF的命名实体识别(NER)模型，使用CONLL 2003数据集。

## 项目结构

```
lab4/
├── README.md              # 项目说明
├── config.py             # 配置文件
├── data_loader.py        # 数据加载器
├── model.py              # LSTM+CRF模型
├── trainer.py            # 训练器
├── evaluate.py           # 评估脚本
├── train.py              # 训练脚本
├── predict.py            # 预测脚本
├── quick_start.py        # 快速开始脚本
├── download_data.py      # 数据下载脚本
├── utils.py              # 工具函数
├── ner/                  # 数据目录
│   ├── eng.train         # 训练数据
│   ├── eng.testa         # 验证数据
│   └── eng.testb         # 测试数据
├── models/               # 保存的模型
└── results/              # 结果文件
```

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- NumPy
- scikit-learn
- matplotlib
- tqdm
- pandas
- seaborn

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

#### 方法一：使用示例数据（推荐用于快速测试）

```bash
python generate_sample_data.py
```

这将生成符合CONLL格式的示例数据，包含人名、组织名、地点名等实体。

#### 方法二：使用真实CONLL 2003数据

```bash
python download_data.py
```

然后按照提示下载和处理CONLL 2003数据集。

### 3. 训练模型

```bash
python train.py
```

### 4. 评估模型

```bash
python evaluate.py
```

### 5. 预测新文本

```bash
# 交互模式
python predict.py --interactive

# 单个文本
python predict.py --text "John Smith works at Microsoft in Seattle."

# 从文件读取
python predict.py --file texts.txt --output predictions.json
```

### 6. 快速演示

```bash
python quick_start.py
```

选择演示训练或预测功能。

## 模型架构

### BiLSTM-CRF模型

- **词嵌入层**: 将词转换为向量表示
- **双向LSTM**: 捕获上下文信息
- **CRF层**: 建模标签间的依赖关系
- **Dropout**: 防止过拟合

### 实体标签

- **PER**: 人名 (Person)
- **ORG**: 组织名 (Organization)  
- **LOC**: 地点名 (Location)
- **MISC**: 其他实体 (Miscellaneous)
- **O**: 非实体

### 标签格式

使用IOB1格式：
- **B-TYPE**: 实体开始 (Beginning)
- **I-TYPE**: 实体内部 (Inside)
- **O**: 非实体 (Outside)

## 评价指标

- **Precision**: 精确率
- **Recall**: 召回率  
- **F1-Score**: F1分数

## 配置参数

在 `config.py` 中可以调整以下参数：

```python
# 模型参数
embedding_dim = 100      # 词嵌入维度
hidden_dim = 200         # LSTM隐藏层维度
num_layers = 2           # LSTM层数
dropout = 0.5            # Dropout率
bidirectional = True     # 是否使用双向LSTM

# 训练参数
batch_size = 32          # 批次大小
learning_rate = 0.001    # 学习率
num_epochs = 50          # 训练轮数
early_stopping_patience = 10  # 早停耐心值
```

## 使用示例

### 训练示例

```python
from config import Config
from data_loader import DataManager
from model import BiLSTMCRF
from trainer import NERTrainer

# 创建配置
config = Config()

# 加载数据
data_manager = DataManager(config)
train_loader, dev_loader, test_loader = data_manager.get_dataloaders()
vocab_info = data_manager.get_vocab_info()

# 创建模型
model = BiLSTMCRF(
    vocab_size=vocab_info['vocab_size'],
    embedding_dim=config.embedding_dim,
    hidden_dim=config.hidden_dim,
    num_layers=config.num_layers,
    num_tags=vocab_info['num_labels'],
    dropout=config.dropout,
    bidirectional=config.bidirectional
)

# 训练模型
trainer = NERTrainer(model, config, device)
results = trainer.train(train_loader, dev_loader, config.num_epochs)
```

### 预测示例

```python
# 加载训练好的模型
model.load_state_dict(torch.load('models/best_model.pth'))

# 预测
text = "John Smith works at Microsoft in Seattle."
words = text.split()
predictions = model.predict(word_ids, lengths)

# 输出结果
for word, pred in zip(words, predictions):
    print(f"{word}: {pred}")
```

## 数据集

### CONLL 2003格式

每行包含4个字段：
```
词 词性 块标签 实体标签
```

示例：
```
John NNP B-NP B-PER
Smith NNP I-NP I-PER
works VBZ B-VP O
at IN B-PP O
Microsoft NNP B-NP B-ORG
```

### 数据文件

- `eng.train`: 训练数据
- `eng.testa`: 验证数据  
- `eng.testb`: 测试数据

## 实验结果

在示例数据上的典型结果：
- **训练集F1**: ~95%
- **验证集F1**: ~90%
- **测试集F1**: ~85%

## 参考论文

1. Bidirectional LSTM-CRF Models for Sequence Tagging
2. Neural Architectures for Named Entity Recognition
3. Enhanced LSTM for Natural Language Inference

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小 `batch_size`
   - 减小 `hidden_dim`

2. **训练不收敛**
   - 调整 `learning_rate`
   - 增加 `num_epochs`
   - 检查数据质量

3. **过拟合**
   - 增加 `dropout`
   - 减少模型复杂度
   - 增加训练数据

### 调试模式

设置 `config.device = "cpu"` 使用CPU进行调试。

## 扩展功能

- 支持预训练词向量（如GloVe）
- 支持字符级特征
- 支持多语言NER
- 支持自定义实体类型



![image-20250703212601728](./image-20250703212601728.png)