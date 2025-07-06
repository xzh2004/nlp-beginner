import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RNNTextClassifier(nn.Module):
    """RNN文本分类模型"""
    
    def __init__(self, config, embedding_layer):
        super(RNNTextClassifier, self).__init__()
        
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_classes = config.num_classes
        self.dropout = config.dropout
        self.bidirectional = config.bidirectional
        
        # Embedding层
        self.embedding = embedding_layer
        
        # RNN层
        self.rnn = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Dropout层
        self.dropout = nn.Dropout(self.dropout)
        
        # 全连接层
        rnn_output_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        self.fc = nn.Linear(rnn_output_dim, self.num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状为 (batch_size, seq_length)
        Returns:
            输出张量，形状为 (batch_size, num_classes)
        """
        # Embedding: (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)
        
        # RNN: (batch_size, seq_length, embedding_dim) -> (batch_size, seq_length, hidden_dim)
        rnn_out, (hidden, cell) = self.rnn(embedded)
        
        # 获取最后一个时间步的输出
        if self.bidirectional:
            # 双向RNN，拼接最后两个隐藏状态
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            # 单向RNN，取最后一层的隐藏状态
            last_hidden = hidden[-1]
        
        # Dropout
        last_hidden = self.dropout(last_hidden)
        
        # 全连接层: (batch_size, hidden_dim) -> (batch_size, num_classes)
        output = self.fc(last_hidden)
        
        return output
    
    def get_attention_weights(self, x):
        """
        获取注意力权重（用于可视化）
        Args:
            x: 输入张量
        Returns:
            注意力权重
        """
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        
        # 简单的注意力机制：使用最后一个隐藏层作为查询
        if self.bidirectional:
            query = torch.cat((rnn_out[:, -1, :self.hidden_dim], rnn_out[:, 0, self.hidden_dim:]), dim=1)
        else:
            query = rnn_out[:, -1, :]
        
        # 计算注意力权重
        attention_weights = torch.bmm(rnn_out, query.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        return attention_weights

class GRUTextClassifier(nn.Module):
    """GRU文本分类模型"""
    
    def __init__(self, config, embedding_layer):
        super(GRUTextClassifier, self).__init__()
        
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_classes = config.num_classes
        self.dropout = config.dropout
        self.bidirectional = config.bidirectional
        
        # Embedding层
        self.embedding = embedding_layer
        
        # GRU层
        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Dropout层
        self.dropout = nn.Dropout(self.dropout)
        
        # 全连接层
        gru_output_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        self.fc = nn.Linear(gru_output_dim, self.num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状为 (batch_size, seq_length)
        Returns:
            输出张量，形状为 (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(x)
        
        # GRU
        gru_out, hidden = self.gru(embedded)
        
        # 获取最后一个时间步的输出
        if self.bidirectional:
            # 双向GRU，拼接最后两个隐藏状态
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            # 单向GRU，取最后一层的隐藏状态
            last_hidden = hidden[-1]
        
        # Dropout
        last_hidden = self.dropout(last_hidden)
        
        # 全连接层
        output = self.fc(last_hidden)
        
        return output

class AttentionRNNTextClassifier(nn.Module):
    """带注意力机制的RNN文本分类模型"""
    
    def __init__(self, config, embedding_layer):
        super(AttentionRNNTextClassifier, self).__init__()
        
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_classes = config.num_classes
        self.dropout = config.dropout
        self.bidirectional = config.bidirectional
        
        # Embedding层
        self.embedding = embedding_layer
        
        # RNN层
        self.rnn = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # 注意力层
        rnn_output_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        self.attention = nn.Linear(rnn_output_dim, 1)
        
        # Dropout层
        self.dropout = nn.Dropout(self.dropout)
        
        # 全连接层
        self.fc = nn.Linear(rnn_output_dim, self.num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        nn.init.xavier_uniform_(self.attention.weight)
        nn.init.constant_(self.attention.bias, 0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状为 (batch_size, seq_length)
        Returns:
            输出张量，形状为 (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(x)
        
        # RNN
        rnn_out, _ = self.rnn(embedded)
        
        # 注意力机制
        attention_weights = torch.softmax(self.attention(rnn_out), dim=1)
        attended_output = torch.sum(attention_weights * rnn_out, dim=1)
        
        # Dropout
        attended_output = self.dropout(attended_output)
        
        # 全连接层
        output = self.fc(attended_output)
        
        return output
    
    def get_attention_weights(self, x):
        """
        获取注意力权重
        Args:
            x: 输入张量
        Returns:
            注意力权重
        """
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        attention_weights = torch.softmax(self.attention(rnn_out), dim=1)
        return attention_weights.squeeze(-1)