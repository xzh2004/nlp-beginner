import json
import re
import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
import nltk

# 下载nltk数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_snli_data(file_path, max_samples=None):
    """加载SNLI数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line.strip())
            if item['gold_label'] != '-':
                data.append({
                    'sentence1': item['sentence1'],
                    'sentence2': item['sentence2'],
                    'label': item['gold_label']
                })
    return data

def preprocess_text(text):
    """文本预处理"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    """分词"""
    return preprocess_text(text).split()

def build_vocab(sentences, min_freq=2):
    """构建词汇表"""
    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
    
    word_freq = Counter()
    for sentence in sentences:
        tokens = tokenize_text(sentence)
        word_freq.update(tokens)
    
    vocab = {}
    idx = 0
    
    for token in special_tokens:
        vocab[token] = idx
        idx += 1
    
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab

def text_to_sequence(text, vocab, max_length=50):
    """将文本转换为序列"""
    tokens = tokenize_text(text)
    sequence = []
    
    sequence.append(vocab.get('<START>', vocab['<UNK>']))
    
    for token in tokens[:max_length-2]:
        sequence.append(vocab.get(token, vocab['<UNK>']))
    
    sequence.append(vocab.get('<END>', vocab['<UNK>']))
    
    while len(sequence) < max_length:
        sequence.append(vocab['<PAD>'])
    
    return sequence[:max_length]

def load_vocab(file_path):
    """加载词汇表"""
    return torch.load(file_path, map_location='cpu')

class SNLIDataset(Dataset):
    """SNLI数据集类"""
    def __init__(self, data, vocab, max_length=50):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        sentence1_seq = text_to_sequence(item['sentence1'], self.vocab, self.max_length)
        sentence2_seq = text_to_sequence(item['sentence2'], self.vocab, self.max_length)
        
        return {
            'sentence1': torch.tensor(sentence1_seq, dtype=torch.long),
            'sentence2': torch.tensor(sentence2_seq, dtype=torch.long),
            'label': item['label']
        }

def collate_fn(batch):
    """数据批处理函数"""
    sentence1_batch = torch.stack([item['sentence1'] for item in batch])
    sentence2_batch = torch.stack([item['sentence2'] for item in batch])
    labels = [item['label'] for item in batch]
    
    return {
        'sentence1': sentence1_batch,
        'sentence2': sentence2_batch,
        'labels': labels
    }

def calculate_accuracy(predictions, labels, label2idx):
    """计算准确率"""
    correct = 0
    total = 0
    
    for pred, label in zip(predictions, labels):
        if pred == label:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

def save_vocab(vocab, file_path):
    """保存词汇表"""
    torch.save(vocab, file_path)

def load_glove_embeddings(glove_path, vocab, embedding_dim=300):
    """加载GloVe预训练词向量"""
    print(f"Loading GloVe embeddings from {glove_path}...")
    
    # 初始化embedding矩阵
    vocab_size = len(vocab)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))
    
    # 加载GloVe词向量
    glove_embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_embeddings[word] = vector
    
    print(f"Loaded {len(glove_embeddings)} GloVe embeddings")
    
    # 为词汇表中的词分配embedding
    found_words = 0
    for word, idx in vocab.items():
        if word in glove_embeddings:
            embedding_matrix[idx] = glove_embeddings[word]
            found_words += 1
        # 特殊token保持随机初始化
        elif word in ['<PAD>', '<UNK>', '<START>', '<END>']:
            if word == '<PAD>':
                embedding_matrix[idx] = np.zeros(embedding_dim)  # PAD token设为0向量
            # 其他特殊token保持随机初始化
    
    print(f"Found {found_words} words in GloVe embeddings")
    return embedding_matrix 