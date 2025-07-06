"""
数据加载器：处理CONLL格式的NER数据
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import (
    load_conll_data, build_vocab, build_label_vocab, 
    sentence_to_ids, sentence_to_labels, pad_sequences
)


class NERDataset(Dataset):
    """
    NER数据集类
    """
    
    def __init__(self, sentences, vocab, label_vocab, config):
        self.sentences = sentences
        self.vocab = vocab
        self.label_vocab = label_vocab
        self.config = config
        
        # 预处理数据
        self.processed_data = self._preprocess_data()
    
    def _preprocess_data(self):
        """
        预处理数据
        """
        processed_data = []
        
        for sentence in self.sentences:
            # 转换为ID序列
            word_ids = sentence_to_ids(sentence, self.vocab, self.config)
            label_ids = sentence_to_labels(sentence, self.label_vocab)
            
            processed_data.append({
                'word_ids': word_ids,
                'label_ids': label_ids,
                'length': len(word_ids)
            })
        
        return processed_data
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        return {
            'word_ids': torch.tensor(item['word_ids'], dtype=torch.long),
            'label_ids': torch.tensor(item['label_ids'], dtype=torch.long),
            'length': item['length']
        }


def collate_fn(batch):
    """
    批处理函数
    """
    word_ids = [item['word_ids'] for item in batch]
    label_ids = [item['label_ids'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    # 填充到相同长度
    max_len = max(lengths)
    
    padded_word_ids = []
    padded_label_ids = []
    masks = []
    
    for word_seq, label_seq, length in zip(word_ids, label_ids, lengths):
        # 填充词ID
        padded_word_seq = torch.cat([
            word_seq,
            torch.zeros(max_len - length, dtype=torch.long)
        ])
        padded_word_ids.append(padded_word_seq)
        
        # 填充标签ID
        padded_label_seq = torch.cat([
            label_seq,
            torch.zeros(max_len - length, dtype=torch.long)
        ])
        padded_label_ids.append(padded_label_seq)
        
        # 创建mask
        mask = torch.cat([
            torch.ones(length, dtype=torch.bool),
            torch.zeros(max_len - length, dtype=torch.bool)
        ])
        masks.append(mask)
    
    return {
        'word_ids': torch.stack(padded_word_ids),
        'label_ids': torch.stack(padded_label_ids),
        'masks': torch.stack(masks),
        'lengths': torch.tensor(lengths, dtype=torch.long)
    }


class DataManager:
    """
    数据管理器
    """
    
    def __init__(self, config):
        self.config = config
        self.vocab = None
        self.label_vocab = None
        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
    
    def load_data(self):
        """
        加载所有数据
        """
        print("Loading CONLL data...")
        
        # 加载原始数据
        train_sentences = load_conll_data(f"{self.config.data_dir}/{self.config.train_file}")
        dev_sentences = load_conll_data(f"{self.config.data_dir}/{self.config.dev_file}")
        test_sentences = load_conll_data(f"{self.config.data_dir}/{self.config.test_file}")
        
        print(f"Loaded {len(train_sentences)} training sentences")
        print(f"Loaded {len(dev_sentences)} dev sentences")
        print(f"Loaded {len(test_sentences)} test sentences")
        
        # 构建词汇表
        print("Building vocabulary...")
        self.vocab = build_vocab(train_sentences, self.config)
        self.label_vocab = build_label_vocab(train_sentences + dev_sentences + test_sentences)
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Label vocabulary size: {len(self.label_vocab)}")
        print(f"Labels: {list(self.label_vocab.keys())}")
        
        # 创建数据集
        self.train_dataset = NERDataset(train_sentences, self.vocab, self.label_vocab, self.config)
        self.dev_dataset = NERDataset(dev_sentences, self.vocab, self.label_vocab, self.config)
        self.test_dataset = NERDataset(test_sentences, self.vocab, self.label_vocab, self.config)
        
        print("Data loading completed!")
    
    def get_dataloaders(self):
        """
        获取数据加载器
        """
        if self.train_dataset is None:
            self.load_data()
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        dev_loader = DataLoader(
            self.dev_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        return train_loader, dev_loader, test_loader
    
    def get_vocab_info(self):
        """
        获取词汇表信息
        """
        return {
            'vocab': self.vocab,
            'label_vocab': self.label_vocab,
            'vocab_size': len(self.vocab),
            'num_labels': len(self.label_vocab)
        } 