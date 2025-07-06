"""
配置文件：定义模型和训练的超参数
"""

class Config:
    # 数据相关
    data_dir = "ner/"
    train_file = "eng.train"
    dev_file = "eng.testa" 
    test_file = "eng.testb"
    
    # 模型参数
    embedding_dim = 100
    hidden_dim = 200
    num_layers = 2
    dropout = 0.5
    bidirectional = True
    
    # 训练参数
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50
    early_stopping_patience = 10
    
    # 词汇表参数
    min_freq = 2
    max_vocab_size = 50000
    
    # 标签相关
    pad_token = "<PAD>"
    unk_token = "<UNK>"
    start_token = "<START>"
    end_token = "<END>"
    
    # 实体标签
    entity_labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    
    # 文件路径
    model_save_dir = "models/"
    results_dir = "results/"
    
    # 设备
    device = "cuda"  # 或 "cpu"
    
    # 随机种子
    random_seed = 42
    
    # 日志
    log_interval = 100
    eval_interval = 500
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value) 