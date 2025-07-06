import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import pickle
import os

class CharDataset(Dataset):
    """字符级数据集类"""
    
    def __init__(self, data, seq_length, unk_idx=None):
        self.data = data
        self.seq_length = seq_length
        self.unk_idx = unk_idx
        
    def __len__(self):
        return len(self.data) - self.seq_length
        
    def __getitem__(self, idx):
        # 输入序列和目标序列
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        # 数据增强：随机mask一些字符
        if self.unk_idx is not None and np.random.random() < 0.1:
            mask_positions = np.random.choice(len(x), size=max(1, len(x)//20), replace=False)
            x = list(x)  # 转为可变
            for pos in mask_positions:
                x[pos] = self.unk_idx
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class CharVocab:
    """字符词汇表类"""
    
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
        
    def build_vocab(self, text):
        """构建词汇表"""
        # 统计字符频率
        char_counts = Counter(text)
        
        # 过滤低频字符
        filtered_chars = [char for char, count in char_counts.items() 
                         if count >= self.min_freq]
        
        # 添加特殊字符
        special_chars = ['<PAD>', '<UNK>', '<START>', '<END>']
        all_chars = special_chars + sorted(filtered_chars)
        
        # 创建映射
        self.char2idx = {char: idx for idx, char in enumerate(all_chars)}
        self.idx2char = {idx: char for idx, char in enumerate(all_chars)}
        self.vocab_size = len(all_chars)
        
        print(f"词汇表大小: {self.vocab_size}")
        print(f"特殊字符: {special_chars}")
        
    def encode(self, text):
        """将文本编码为索引序列"""
        return [self.char2idx.get(char, self.char2idx['<UNK>']) for char in text]
    
    def decode(self, indices):
        """将索引序列解码为文本"""
        return ''.join([self.idx2char.get(idx, '<UNK>') for idx in indices])
    
    def save_vocab(self, filepath):
        """保存词汇表"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'char2idx': self.char2idx,
                'idx2char': self.idx2char,
                'vocab_size': self.vocab_size,
                'min_freq': self.min_freq
            }, f)
    
    def load_vocab(self, filepath):
        """加载词汇表"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.char2idx = data['char2idx']
            self.idx2char = data['idx2char']
            self.vocab_size = data['vocab_size']
            self.min_freq = data['min_freq']

def load_data(filepath, seq_length=50, batch_size=32, min_freq=1, 
              train_ratio=0.8, val_ratio=0.1, vocab_path=None):
    """
    加载和预处理数据
    
    Args:
        filepath: 数据文件路径
        seq_length: 序列长度
        batch_size: 批次大小
        min_freq: 最小字符频率
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        vocab_path: 词汇表保存路径
    
    Returns:
        train_loader, val_loader, test_loader, vocab
    """
    
    # 读取文本数据
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"原始文本长度: {len(text)} 字符")
    print(f"文本前100个字符: {text[:100]}")
    
    # 创建词汇表
    vocab = CharVocab(min_freq=min_freq)
    
    if vocab_path and os.path.exists(vocab_path):
        print(f"加载已有词汇表: {vocab_path}")
        vocab.load_vocab(vocab_path)
    else:
        print("构建新词汇表...")
        vocab.build_vocab(text)
        if vocab_path:
            vocab.save_vocab(vocab_path)
    
    # 编码文本
    encoded_text = vocab.encode(text)
    print(f"编码后序列长度: {len(encoded_text)}")
    
    # 划分数据集
    total_len = len(encoded_text)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    
    train_data = encoded_text[:train_len]
    val_data = encoded_text[train_len:train_len + val_len]
    test_data = encoded_text[train_len + val_len:]
    
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    # 创建数据集
    unk_idx = vocab.char2idx.get('<UNK>', 1)
    train_dataset = CharDataset(train_data, seq_length, unk_idx=unk_idx)
    val_dataset = CharDataset(val_data, seq_length, unk_idx=unk_idx)
    test_dataset = CharDataset(test_data, seq_length, unk_idx=unk_idx)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, drop_last=True)
    
    return train_loader, val_loader, test_loader, vocab 