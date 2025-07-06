import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSTMLanguageModel(nn.Module):
    """基于LSTM的语言模型"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.2):
        super(LSTMLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 字符嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # 初始化权重
        self.init_weights()
        
    def init_weights(self):
        """初始化模型权重"""
        init_range = 0.1
        
        # 初始化嵌入层
        self.embedding.weight.data.uniform_(-init_range, init_range)
        
        # 初始化LSTM权重
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # 初始化输出层
        self.output_layer.weight.data.uniform_(-init_range, init_range)
        self.output_layer.bias.data.zero_()
        
    def forward(self, x, hidden=None):
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_length]
            hidden: 初始隐藏状态 (h0, c0)
        
        Returns:
            output: 输出概率 [batch_size, seq_length, vocab_size]
            hidden: 最终隐藏状态 (hn, cn)
        """
        batch_size, seq_length = x.size()
        
        # 字符嵌入
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        
        # LSTM前向传播
        if hidden is None:
            # 初始化隐藏状态
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            hidden = (h0, c0)
        
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # 输出层
        output = self.output_layer(lstm_out)  # [batch_size, seq_length, vocab_size]
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

class GRULanguageModel(nn.Module):
    """基于GRU的语言模型"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.2):
        super(GRULanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 字符嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # GRU层
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # 初始化权重
        self.init_weights()
        
    def init_weights(self):
        """初始化模型权重"""
        init_range = 0.1
        
        # 初始化嵌入层
        self.embedding.weight.data.uniform_(-init_range, init_range)
        
        # 初始化GRU权重
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # 初始化输出层
        self.output_layer.weight.data.uniform_(-init_range, init_range)
        self.output_layer.bias.data.zero_()
        
    def forward(self, x, hidden=None):
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_length]
            hidden: 初始隐藏状态 h0
        
        Returns:
            output: 输出概率 [batch_size, seq_length, vocab_size]
            hidden: 最终隐藏状态 hn
        """
        batch_size, seq_length = x.size()
        
        # 字符嵌入
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        
        # GRU前向传播
        if hidden is None:
            # 初始化隐藏状态
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        gru_out, hidden = self.gru(embedded, hidden)
        
        # Dropout
        gru_out = self.dropout(gru_out)
        
        # 输出层
        output = self.output_layer(gru_out)  # [batch_size, seq_length, vocab_size]
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_perplexity(model, data_loader, device, vocab):
    """
    计算困惑度
    
    Args:
        model: 语言模型
        data_loader: 数据加载器
        device: 设备
        vocab: 词汇表
    
    Returns:
        perplexity: 困惑度
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            
            # 前向传播
            output, _ = model(x)
            
            # 计算损失
            loss = F.cross_entropy(
                output.view(-1, vocab.vocab_size), 
                y.view(-1),
                ignore_index=vocab.char2idx.get('<PAD>', -1)
            )
            
            total_loss += loss.item() * x.size(0) * x.size(1)
            total_tokens += x.size(0) * x.size(1)
    
    # 计算平均损失和困惑度
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss 