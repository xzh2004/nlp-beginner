import torch

class Config:
    # 数据相关
    train_path = "snli_1.0/snli_1.0_train.jsonl"
    dev_path = "snli_1.0/snli_1.0_dev.jsonl"
    test_path = "snli_1.0/snli_1.0_test.jsonl"
    
    # GloVe预训练词向量路径
    glove_path = "../lab2/glove.6B/glove.6B.300d.txt"
    
    # 模型参数
    embedding_dim = 300
    hidden_size = 300
    num_classes = 3
    dropout = 0.5
    max_length = 50
    min_freq = 2
    
    # 训练参数
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 词汇表相关
    pad_token = '<PAD>'
    unk_token = '<UNK>'
    start_token = '<START>'
    end_token = '<END>'
    
    # 标签映射
    label2idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    idx2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    
    # 保存路径
    model_save_path = "models/esim_model.pth"
    vocab_save_path = "models/vocab.pth"
    results_save_path = "results/" 