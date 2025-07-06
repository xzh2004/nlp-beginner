"""
Hugging Face数据集加载器：处理CONLL 2003数据集
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
from collections import Counter
from utils import set_random_seed


class HFNERDataset(Dataset):
    """
    基于Hugging Face数据集的NER数据集类
    """
    
    def __init__(self, hf_dataset, vocab, label_vocab, config):
        self.hf_dataset = hf_dataset
        self.vocab = vocab
        self.label_vocab = label_vocab
        self.config = config
        
        # 预处理数据
        self.processed_data = self._preprocess_data()
    
    def _preprocess_data(self):
        """
        预处理Hugging Face数据集
        """
        processed_data = []
        
        for example in self.hf_dataset:
            tokens = example['tokens']
            ner_tags = example['ner_tags']
            
            # 转换为ID序列
            word_ids = [self.vocab.get(self.config.start_token, self.vocab[self.config.unk_token])]
            
            for token in tokens:
                word_id = self.vocab.get(token.lower(), self.vocab[self.config.unk_token])
                word_ids.append(word_id)
            
            word_ids.append(self.vocab.get(self.config.end_token, self.vocab[self.config.unk_token]))
            
            # 转换标签
            label_ids = [self.label_vocab['O']]  # START标签
            for tag_id in ner_tags:
                # 将数字标签转换为字符串标签
                tag_name = self._id_to_tag(tag_id)
                label_ids.append(self.label_vocab[tag_name])
            label_ids.append(self.label_vocab['O'])  # END标签
            
            processed_data.append({
                'word_ids': word_ids,
                'label_ids': label_ids,
                'length': len(word_ids)
            })
        
        return processed_data
    
    def _id_to_tag(self, tag_id):
        """
        将数字标签ID转换为字符串标签
        """
        tag_mapping = {
            0: 'O',
            1: 'B-PER',
            2: 'I-PER', 
            3: 'B-ORG',
            4: 'I-ORG',
            5: 'B-LOC',
            6: 'I-LOC',
            7: 'B-MISC',
            8: 'I-MISC'
        }
        return tag_mapping.get(tag_id, 'O')
    
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


class HFDataManager:
    """
    基于Hugging Face的数据管理器
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
        加载Hugging Face CONLL 2003数据集
        """
        print("Loading CONLL 2003 dataset from Hugging Face...")
        
        # 加载数据集
        dataset = load_dataset("conll2003", trust_remote_code=True)
        
        train_data = dataset['train']
        validation_data = dataset['validation']  # 注意：HF数据集用'validation'而不是'dev'
        test_data = dataset['test']
        
        print(f"Loaded {len(train_data)} training sentences")
        print(f"Loaded {len(validation_data)} validation sentences")
        print(f"Loaded {len(test_data)} test sentences")
        
        # 构建词汇表
        print("Building vocabulary...")
        self.vocab = self._build_vocab(train_data)
        self.label_vocab = self._build_label_vocab(train_data)
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Label vocabulary size: {len(self.label_vocab)}")
        print(f"Labels: {list(self.label_vocab.keys())}")
        
        # 创建数据集
        self.train_dataset = HFNERDataset(train_data, self.vocab, self.label_vocab, self.config)
        self.dev_dataset = HFNERDataset(validation_data, self.vocab, self.label_vocab, self.config)
        self.test_dataset = HFNERDataset(test_data, self.vocab, self.label_vocab, self.config)
        
        print("Data loading completed!")
    
    def _build_vocab(self, train_data):
        """
        构建词汇表
        """
        word_freq = Counter()
        
        for example in train_data:
            for token in example['tokens']:
                word_freq[token.lower()] += 1
        
        # 构建词汇表
        vocab = {self.config.pad_token: 0, self.config.unk_token: 1, 
                self.config.start_token: 2, self.config.end_token: 3}
        
        # 按频率排序，只保留频率>=min_freq的词
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        for word, freq in sorted_words:
            if freq >= self.config.min_freq and len(vocab) < self.config.max_vocab_size:
                vocab[word] = len(vocab)
        
        return vocab
    
    def _build_label_vocab(self, train_data):
        """
        构建标签词汇表
        """
        label_set = set()
        for example in train_data:
            for tag_id in example['ner_tags']:
                tag_name = self._id_to_tag(tag_id)
                label_set.add(tag_name)
        
        label_vocab = {label: idx for idx, label in enumerate(sorted(label_set))}
        return label_vocab
    
    def _id_to_tag(self, tag_id):
        """
        将数字标签ID转换为字符串标签
        """
        tag_mapping = {
            0: 'O',
            1: 'B-PER',
            2: 'I-PER', 
            3: 'B-ORG',
            4: 'I-ORG',
            5: 'B-LOC',
            6: 'I-LOC',
            7: 'B-MISC',
            8: 'I-MISC'
        }
        return tag_mapping.get(tag_id, 'O')
    
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