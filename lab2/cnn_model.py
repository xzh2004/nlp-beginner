import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNTextClassifier(nn.Module):
    """CNN文本分类模型，参考论文 'Convolutional Neural Networks for Sentence Classification'"""
    
    def __init__(self, config, embedding_layer):
        super(CNNTextClassifier, self).__init__()
        
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.num_filters = config.num_filters
        self.filter_sizes = config.filter_sizes
        self.num_classes = config.num_classes
        self.dropout = config.dropout
        
        # Embedding层
        self.embedding = embedding_layer
        
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, (k, self.embedding_dim), padding=(k//2, 0))
            for k in self.filter_sizes
        ])
        
        # Dropout层
        self.dropout = nn.Dropout(self.dropout)
        
        # 全连接层
        self.fc = nn.Linear(len(self.filter_sizes) * self.num_filters, self.num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)
        
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
        
        # 添加通道维度: (batch_size, seq_length, embedding_dim) -> (batch_size, 1, seq_length, embedding_dim)
        embedded = embedded.unsqueeze(1)
        
        # 应用卷积和池化
        conv_outputs = []
        for conv in self.convs:
            # 卷积: (batch_size, 1, seq_length, embedding_dim) -> (batch_size, num_filters, seq_length, 1)
            conv_out = conv(embedded)
            
            # ReLU激活
            conv_out = F.relu(conv_out)
            
            # 移除最后一个维度: (batch_size, num_filters, seq_length, 1) -> (batch_size, num_filters, seq_length)
            conv_out = conv_out.squeeze(-1)
            
            # 最大池化: (batch_size, num_filters, seq_length) -> (batch_size, num_filters)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            pooled = pooled.squeeze(-1)
            
            conv_outputs.append(pooled)
        
        # 拼接所有卷积输出: (batch_size, num_filters * len(filter_sizes))
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Dropout
        concatenated = self.dropout(concatenated)
        
        # 全连接层: (batch_size, num_filters * len(filter_sizes)) -> (batch_size, num_classes)
        output = self.fc(concatenated)
        
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
        embedded = embedded.unsqueeze(1)
        
        attention_weights = []
        for conv in self.convs:
            conv_out = conv(embedded)
            conv_out = F.relu(conv_out)
            conv_out = conv_out.squeeze(-1)
            
            # 计算注意力权重
            attention = F.softmax(conv_out, dim=2)
            attention_weights.append(attention)
        
        return attention_weights

class CNNTextClassifierWithBatchNorm(nn.Module):
    """带BatchNorm的CNN文本分类模型"""
    
    def __init__(self, config, embedding_layer):
        super(CNNTextClassifierWithBatchNorm, self).__init__()
        
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.num_filters = config.num_filters
        self.filter_sizes = config.filter_sizes
        self.num_classes = config.num_classes
        self.dropout = config.dropout
        
        # Embedding层
        self.embedding = embedding_layer
        
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, (k, self.embedding_dim), padding=(k//2, 0))
            for k in self.filter_sizes
        ])
        
        # BatchNorm层
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(self.num_filters)
            for _ in self.filter_sizes
        ])
        
        # Dropout层
        self.dropout = nn.Dropout(self.dropout)
        
        # 全连接层
        self.fc = nn.Linear(len(self.filter_sizes) * self.num_filters, self.num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)
        
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
        embedded = embedded.unsqueeze(1)
        
        # 应用卷积、BatchNorm和池化
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            # 卷积
            conv_out = conv(embedded)
            
            # BatchNorm
            conv_out = bn(conv_out)
            
            # ReLU激活
            conv_out = F.relu(conv_out)
            
            # 移除最后一个维度
            conv_out = conv_out.squeeze(-1)
            
            # 最大池化
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            pooled = pooled.squeeze(-1)
            
            conv_outputs.append(pooled)
        
        # 拼接所有卷积输出
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Dropout
        concatenated = self.dropout(concatenated)
        
        # 全连接层
        output = self.fc(concatenated)
        
        return output