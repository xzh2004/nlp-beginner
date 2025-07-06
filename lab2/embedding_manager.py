import numpy as np
import torch
import torch.nn as nn
import os

class EmbeddingManager:
    """Embedding管理器，支持不同的初始化方式"""
    
    def __init__(self, config, word_to_idx):
        self.config = config
        self.word_to_idx = word_to_idx
        self.vocab_size = len(word_to_idx)
        self.embedding_dim = config.embedding_dim
        
    def load_glove_embeddings(self):
        """加载GloVe预训练embedding"""
        print(f"加载GloVe {self.config.glove_dim}d embeddings...")
        
        glove_file = os.path.join(self.config.glove_dir, f'glove.6B.{self.config.glove_dim}d.txt')
        
        if not os.path.exists(glove_file):
            raise FileNotFoundError(f"GloVe文件不存在: {glove_file}")
        
        # 加载GloVe embeddings
        glove_embeddings = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                glove_embeddings[word] = vector
        
        print(f"加载了 {len(glove_embeddings)} 个GloVe embeddings")
        
        # 当使用GloVe时，embedding_dim应该与glove_dim一致
        actual_embedding_dim = self.config.glove_dim
        if self.embedding_dim != actual_embedding_dim:
            print(f"警告: embedding_dim ({self.embedding_dim}) 与 glove_dim ({actual_embedding_dim}) 不一致")
            print(f"自动调整embedding_dim为 {actual_embedding_dim}")
            self.embedding_dim = actual_embedding_dim
        
        # 创建embedding矩阵
        embedding_matrix = np.random.normal(scale=0.6, size=(self.vocab_size, self.embedding_dim))
        
        # 用GloVe embeddings初始化
        found_words = 0
        for word, idx in self.word_to_idx.items():
            if word in glove_embeddings:
                embedding_matrix[idx] = glove_embeddings[word]
                found_words += 1
        
        print(f"在词汇表中找到 {found_words}/{self.vocab_size} 个词的GloVe embeddings")
        
        return embedding_matrix
    
    def create_random_embeddings(self):
        """创建随机初始化的embeddings"""
        print("创建随机初始化的embeddings...")
        
        # 使用正态分布初始化
        embedding_matrix = np.random.normal(scale=0.6, size=(self.vocab_size, self.embedding_dim))
        
        # 将特殊标记的embedding设为0
        if '<PAD>' in self.word_to_idx:
            embedding_matrix[self.word_to_idx['<PAD>']] = 0
        
        return embedding_matrix
    
    def create_word2vec_style_embeddings(self):
        """创建Word2Vec风格的embeddings（使用Xavier初始化）"""
        print("创建Word2Vec风格的embeddings...")
        
        # 使用Xavier初始化
        embedding_matrix = np.random.uniform(
            low=-np.sqrt(3.0 / self.embedding_dim),
            high=np.sqrt(3.0 / self.embedding_dim),
            size=(self.vocab_size, self.embedding_dim)
        )
        
        # 将特殊标记的embedding设为0
        if '<PAD>' in self.word_to_idx:
            embedding_matrix[self.word_to_idx['<PAD>']] = 0
        
        return embedding_matrix
    
    def get_embedding_matrix(self):
        """根据配置获取embedding矩阵"""
        if self.config.embedding == 'glove':
            embedding_matrix = self.load_glove_embeddings()
        elif self.config.embedding == 'word2vec':
            embedding_matrix = self.create_word2vec_style_embeddings()
        else:  # random
            embedding_matrix = self.create_random_embeddings()
        
        return embedding_matrix
    
    def create_embedding_layer(self, trainable=True):
        """创建embedding层"""
        embedding_matrix = self.get_embedding_matrix()
        
        embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.word_to_idx.get('<PAD>', 0))
        embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        if not trainable:
            embedding_layer.weight.requires_grad = False
        
        return embedding_layer