import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class ESIM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, dropout=0.5, pretrained_embeddings=None):
        super(ESIM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 如果提供了预训练embedding，则初始化
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            print("Initialized embedding layer with pre-trained GloVe vectors")
        
        # 输入编码层 - BiLSTM
        self.input_encoder = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            bidirectional=True, 
            batch_first=True,
            dropout=dropout if 1 > 1 else 0
        )
        
        # 注意力层
        self.attention = nn.Linear(hidden_size * 2, hidden_size * 2)
        
        # 推理层 - BiLSTM
        self.inference_encoder = nn.LSTM(
            hidden_size * 8,  # 组合后的维度
            hidden_size, 
            bidirectional=True, 
            batch_first=True,
            dropout=dropout if 1 > 1 else 0
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 8, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, sentence1, sentence2):
        batch_size = sentence1.size(0)
        
        # 1. 词嵌入
        sentence1_embedded = self.embedding(sentence1)  # [batch, seq1_len, embedding_dim]
        sentence2_embedded = self.embedding(sentence2)  # [batch, seq2_len, embedding_dim]
        
        # 2. 输入编码 - BiLSTM
        sentence1_encoded, _ = self.input_encoder(sentence1_embedded)  # [batch, seq1_len, hidden_size*2]
        sentence2_encoded, _ = self.input_encoder(sentence2_embedded)  # [batch, seq2_len, hidden_size*2]
        
        # 3. 注意力交互
        # 计算注意力权重矩阵
        attention_weights = torch.bmm(sentence1_encoded, sentence2_encoded.transpose(1, 2))  # [batch, seq1_len, seq2_len]
        
        # 软对齐
        sentence1_aligned = torch.bmm(F.softmax(attention_weights, dim=-1), sentence2_encoded)  # [batch, seq1_len, hidden_size*2]
        sentence2_aligned = torch.bmm(F.softmax(attention_weights.transpose(1, 2), dim=-1), sentence1_encoded)  # [batch, seq2_len, hidden_size*2]
        
        # 4. 组合特征
        sentence1_combined = torch.cat([
            sentence1_encoded,
            sentence1_aligned,
            sentence1_encoded - sentence1_aligned,
            sentence1_encoded * sentence1_aligned
        ], dim=-1)  # [batch, seq1_len, hidden_size*8]
        
        sentence2_combined = torch.cat([
            sentence2_encoded,
            sentence2_aligned,
            sentence2_encoded - sentence2_aligned,
            sentence2_encoded * sentence2_aligned
        ], dim=-1)  # [batch, seq2_len, hidden_size*8]
        
        # 5. 推理层 - BiLSTM
        sentence1_inferred, _ = self.inference_encoder(sentence1_combined)  # [batch, seq1_len, hidden_size*2]
        sentence2_inferred, _ = self.inference_encoder(sentence2_combined)  # [batch, seq2_len, hidden_size*2]
        
        # 6. 池化
        # 最大池化
        sentence1_max_pooled = torch.max(sentence1_inferred, dim=1)[0]  # [batch, hidden_size*2]
        sentence2_max_pooled = torch.max(sentence2_inferred, dim=1)[0]  # [batch, hidden_size*2]
        
        # 平均池化
        sentence1_avg_pooled = torch.mean(sentence1_inferred, dim=1)  # [batch, hidden_size*2]
        sentence2_avg_pooled = torch.mean(sentence2_inferred, dim=1)  # [batch, hidden_size*2]
        
        # 拼接池化结果
        sentence1_pooled = torch.cat([sentence1_max_pooled, sentence1_avg_pooled], dim=-1)  # [batch, hidden_size*4]
        sentence2_pooled = torch.cat([sentence2_max_pooled, sentence2_avg_pooled], dim=-1)  # [batch, hidden_size*4]
        
        # 7. 最终表示
        final_representation = torch.cat([sentence1_pooled, sentence2_pooled], dim=-1)  # [batch, hidden_size*8]
        
        # 8. 分类
        output = self.output_layer(final_representation)  # [batch, num_classes]
        
        return output

class ESIMWithDropout(nn.Module):
    """带dropout的ESIM模型"""
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, dropout=0.5, pretrained_embeddings=None):
        super(ESIMWithDropout, self).__init__()
        
        self.esim = ESIM(vocab_size, embedding_dim, hidden_size, num_classes, dropout, pretrained_embeddings)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, sentence1, sentence2):
        # 在训练时应用dropout
        if self.training:
            sentence1 = self.dropout(sentence1)
            sentence2 = self.dropout(sentence2)
        
        return self.esim(sentence1, sentence2) 