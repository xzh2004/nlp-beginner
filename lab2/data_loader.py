import pandas as pd
import numpy as np
import re
import pickle
import os
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

class TextDataset(Dataset):
    """文本数据集类"""
    def __init__(self, texts, labels, word_to_idx, max_length):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 将文本转换为索引序列
        indices = self.text_to_indices(text)
        
        return {
            'text': torch.LongTensor(indices),
            'label': torch.LongTensor([label])
        }
    
    def text_to_indices(self, text):
        """将文本转换为索引序列"""
        words = text.split()
        indices = []
        
        for word in words[:self.max_length]:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx['<UNK>'])
        
        # 填充到最大长度
        while len(indices) < self.max_length:
            indices.append(self.word_to_idx['<PAD>'])
        
        return indices

class DataProcessor:
    """数据处理器"""
    def __init__(self, config):
        self.config = config
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq = Counter()
        
    def preprocess_text(self, text):
        """文本预处理"""
        if pd.isna(text):
            return ""
        
        # 转换为小写
        text = text.lower()
        
        # 去除特殊字符，保留字母、数字和空格
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def build_vocabulary(self, texts, min_freq=2):
        """构建词汇表"""
        print("构建词汇表...")
        
        # 统计词频
        for text in texts:
            words = text.split()
            self.word_freq.update(words)
        
        # 过滤低频词
        vocab = ['<PAD>', '<UNK>']  # 特殊标记
        for word, freq in self.word_freq.most_common():
            if freq >= min_freq:
                vocab.append(word)
        
        # 创建映射
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"词汇表大小: {len(vocab)}")
        
        return vocab
    
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        
        # 加载训练数据（包含标签）
        train_file = os.path.join(self.config.data_dir, 'train.tsv')
        train_data = pd.read_csv(train_file, sep='\t')
        
        # 加载测试数据（不包含标签）
        test_file = os.path.join(self.config.data_dir, 'test.tsv')
        test_data = pd.read_csv(test_file, sep='\t')
        
        print(f"训练数据形状: {train_data.shape} (包含标签)")
        print(f"测试数据形状: {test_data.shape} (不包含标签)")
        
        return train_data, test_data
    
    def process_data(self, train_data, test_data):
        """处理数据
        
        注意：这里我们将训练数据划分为训练集、验证集和测试集
        原始的test.tsv文件是真正的测试集，没有标签，通常用于竞赛提交
        """
        print("处理数据...")
        print("注意：将训练数据划分为训练集(70%)、验证集(15%)、测试集(15%)")
        print("原始test.tsv文件是真正的测试集，没有标签")
        
        # 预处理文本
        train_data['processed_text'] = train_data['Phrase'].apply(self.preprocess_text)
        test_data['processed_text'] = test_data['Phrase'].apply(self.preprocess_text)
        
        # 过滤空文本
        train_data = train_data[train_data['processed_text'].str.len() > 0]
        test_data = test_data[test_data['processed_text'].str.len() > 0]
        
        # 构建词汇表（只使用训练数据）
        all_texts = train_data['processed_text'].tolist()
        vocab = self.build_vocabulary(all_texts, self.config.min_word_freq)
        
        # 将训练数据划分为训练集、验证集、测试集
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_data['processed_text'].values,
            train_data['Sentiment'].values,
            test_size=self.config.val_ratio + self.config.test_ratio,
            random_state=self.config.random_seed,
            stratify=train_data['Sentiment']
        )
        
        # 进一步划分验证集和测试集
        val_ratio_adjusted = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            val_texts, val_labels,
            test_size=1-val_ratio_adjusted,
            random_state=self.config.random_seed,
            stratify=val_labels
        )
        
        print(f"训练集大小: {len(train_texts)} (用于模型训练)")
        print(f"验证集大小: {len(val_texts)} (用于超参数调优)")
        print(f"测试集大小: {len(test_texts)} (用于最终评估)")
        print(f"原始测试集大小: {len(test_data)} (无标签，用于竞赛提交)")
        
        # 创建数据集
        train_dataset = TextDataset(train_texts, train_labels, self.word_to_idx, self.config.max_seq_length)
        val_dataset = TextDataset(val_texts, val_labels, self.word_to_idx, self.config.max_seq_length)
        test_dataset = TextDataset(test_texts, test_labels, self.word_to_idx, self.config.max_seq_length)
        
        # 创建数据加载器
        train_loader = TorchDataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = TorchDataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def save_vocabulary(self, filepath):
        """保存词汇表"""
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'word_freq': self.word_freq
            }, f)
        
        print(f"词汇表已保存到: {filepath}")
    
    def load_vocabulary(self, filepath):
        """加载词汇表"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.word_to_idx = data['word_to_idx']
            self.idx_to_word = data['idx_to_word']
            self.word_freq = data['word_freq']
        
        print(f"词汇表已从 {filepath} 加载")
        print(f"词汇表大小: {len(self.word_to_idx)}")
    
    def text_to_indices(self, text):
        """将文本转换为索引序列"""
        words = text.split()
        indices = []
        
        for word in words[:self.config.max_seq_length]:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx['<UNK>'])
        
        # 填充到最大长度
        while len(indices) < self.config.max_seq_length:
            indices.append(self.word_to_idx['<PAD>'])
        
        return indices