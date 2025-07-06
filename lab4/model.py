"""
LSTM+CRF模型：实现命名实体识别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CRF(nn.Module):
    """
    条件随机场层
    """
    
    def __init__(self, num_tags, start_tag_id, end_tag_id, pad_tag_id=0):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.start_tag_id = start_tag_id
        self.end_tag_id = end_tag_id
        self.pad_tag_id = pad_tag_id
        
        # 转移矩阵：从标签i到标签j的转移分数
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # 初始化转移矩阵
        self._init_transitions()
    
    def _init_transitions(self):
        """
        初始化转移矩阵
        """
        # 从START标签的转移
        self.transitions.data[self.start_tag_id, :] = -10000
        self.transitions.data[self.start_tag_id, self.start_tag_id] = 0
        
        # 到END标签的转移
        self.transitions.data[:, self.end_tag_id] = -10000
        self.transitions.data[self.end_tag_id, self.end_tag_id] = 0
        
        # PAD标签的转移
        self.transitions.data[self.pad_tag_id, :] = -10000
        self.transitions.data[:, self.pad_tag_id] = -10000
        self.transitions.data[self.pad_tag_id, self.pad_tag_id] = 0
    
    def _forward_alg(self, emissions, mask):
        """
        前向算法：计算所有可能路径的分数
        """
        batch_size, seq_len, num_tags = emissions.size()
        
        # 初始化alpha矩阵
        alpha = torch.full((batch_size, num_tags), -10000, device=emissions.device)
        alpha[:, self.start_tag_id] = 0
        
        # 逐时间步计算
        for t in range(seq_len):
            # 当前时间步的mask
            current_mask = mask[:, t].unsqueeze(1)  # [batch_size, 1]
            
            # 发射分数
            emit_scores = emissions[:, t, :]  # [batch_size, num_tags]
            
            # 转移分数
            trans_scores = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            
            # 计算新的alpha
            next_alpha = alpha.unsqueeze(2) + trans_scores + emit_scores.unsqueeze(1)
            next_alpha = torch.logsumexp(next_alpha, dim=1)  # [batch_size, num_tags]
            
            # 更新alpha（只在有效位置更新）
            alpha = torch.where(current_mask, next_alpha, alpha)
        
        # 最后一步转移到END标签
        alpha = alpha + self.transitions[:, self.end_tag_id]
        return torch.logsumexp(alpha, dim=1)  # [batch_size]
    
    def _score_sequence(self, emissions, tags, mask):
        """
        计算给定标签序列的分数
        """
        batch_size, seq_len, num_tags = emissions.size()
        
        # 获取发射分数
        emit_scores = emissions.gather(2, tags.unsqueeze(2)).squeeze(2)  # [batch_size, seq_len]
        
        # 获取转移分数
        start_tags = tags[:, 0]  # [batch_size]
        end_tags = tags[:, -1]   # [batch_size]
        
        # 第一个标签的转移分数（从START）
        trans_scores = self.transitions[self.start_tag_id, start_tags]  # [batch_size]
        
        # 中间标签的转移分数
        for t in range(seq_len - 1):
            current_tags = tags[:, t]
            next_tags = tags[:, t + 1]
            trans_scores += self.transitions[current_tags, next_tags] * mask[:, t + 1]
        
        # 最后一个标签的转移分数（到END）
        trans_scores += self.transitions[end_tags, self.end_tag_id] * mask[:, -1]
        
        # 总分数
        total_scores = emit_scores.sum(dim=1) + trans_scores
        return total_scores
    
    def _viterbi_decode(self, emissions, mask):
        """
        Viterbi算法：找到最优标签序列
        """
        batch_size, seq_len, num_tags = emissions.size()
        
        # 初始化
        scores = torch.full((batch_size, num_tags), -10000, device=emissions.device)
        scores[:, self.start_tag_id] = 0
        history = []
        
        # 逐时间步计算
        for t in range(seq_len):
            current_mask = mask[:, t].unsqueeze(1)
            
            emit_scores = emissions[:, t, :]
            trans_scores = self.transitions.unsqueeze(0)
            
            # 计算所有可能的前驱状态
            next_scores = scores.unsqueeze(2) + trans_scores + emit_scores.unsqueeze(1)
            next_scores, indices = torch.max(next_scores, dim=1)
            
            # 更新scores和history
            scores = torch.where(current_mask, next_scores, scores)
            history.append(indices)
        
        # 最后一步转移到END标签
        scores = scores + self.transitions[:, self.end_tag_id]
        best_scores, best_tags = torch.max(scores, dim=1)
        
        # 回溯找到最优路径
        best_paths = [best_tags]
        for t in range(seq_len - 1, -1, -1):
            best_tags = history[t].gather(1, best_tags.unsqueeze(1)).squeeze(1)
            best_paths.append(best_tags)
        
        best_paths.reverse()
        best_paths = torch.stack(best_paths, dim=1)
        
        return best_scores, best_paths
    
    def forward(self, emissions, mask, tags=None):
        """
        前向传播
        """
        if tags is None:
            # 推理模式：使用Viterbi算法
            best_scores, best_paths = self._viterbi_decode(emissions, mask)
            return best_scores, best_paths
        else:
            # 训练模式：计算负对数似然
            forward_scores = self._forward_alg(emissions, mask)
            gold_scores = self._score_sequence(emissions, tags, mask)
            # 返回 batch loss 的和，保证loss.backward()可用
            return (forward_scores - gold_scores).sum()


class BiLSTMCRF(nn.Module):
    """
    双向LSTM + CRF模型
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, 
                 num_tags, dropout=0.5, bidirectional=True, pad_idx=0):
        super(BiLSTMCRF, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_tags = num_tags
        self.bidirectional = bidirectional
        self.pad_idx = pad_idx
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 输出层
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.hidden2tag = nn.Linear(lstm_output_dim, num_tags)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # CRF层
        self.crf = CRF(num_tags, start_tag_id=2, end_tag_id=3, pad_tag_id=0)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化模型权重
        """
        # 初始化嵌入层
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        nn.init.constant_(self.embedding.weight[self.pad_idx], 0)
        
        # 初始化LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # 初始化输出层
        nn.init.xavier_normal_(self.hidden2tag.weight)
        nn.init.constant_(self.hidden2tag.bias, 0)
    
    def _get_lstm_features(self, word_ids, lengths):
        """
        获取LSTM特征
        """
        # 词嵌入
        embedded = self.embedding(word_ids)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # Pack序列
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(packed)
        
        # Unpack序列
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # 输出层
        emissions = self.hidden2tag(lstm_out)  # [batch_size, seq_len, num_tags]
        
        return emissions
    
    def forward(self, word_ids, lengths, labels=None, mask=None):
        """
        前向传播
        """
        # 获取LSTM特征
        emissions = self._get_lstm_features(word_ids, lengths)
        
        # 如果没有提供mask，根据lengths创建
        if mask is None:
            mask = torch.arange(emissions.size(1), device=emissions.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        if labels is not None:
            # 训练模式：计算损失
            loss = self.crf(emissions, mask, labels)
            return loss
        else:
            # 推理模式：预测标签
            scores, tag_seq = self.crf(emissions, mask)
            return scores, tag_seq
    
    def predict(self, word_ids, lengths):
        """
        预测标签序列
        """
        self.eval()
        with torch.no_grad():
            emissions = self._get_lstm_features(word_ids, lengths)
            mask = torch.arange(emissions.size(1), device=emissions.device).unsqueeze(0) < lengths.unsqueeze(1)
            scores, tag_seq = self.crf(emissions, mask)
            return tag_seq 