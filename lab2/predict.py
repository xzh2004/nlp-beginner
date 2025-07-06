 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import pickle
import re
import argparse
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from cnn_model import CNNTextClassifier
from rnn_model import RNNTextClassifier
from data_loader import DataProcessor

class TextPredictor:
    """文本预测器"""
    
    def __init__(self, model_path, vocab_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载词汇表
        self.load_vocabulary(vocab_path)
        
        # 加载模型
        self.load_model(model_path)
        
        # 情感标签映射
        self.sentiment_labels = {
            0: "非常负面",
            1: "负面", 
            2: "中性",
            3: "正面",
            4: "非常正面"
        }
    
    def load_vocabulary(self, vocab_path):
        """加载词汇表"""
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
            self.word_to_idx = vocab_data['word_to_idx']
            self.idx_to_word = vocab_data['idx_to_word']
        
        print(f"词汇表大小: {len(self.word_to_idx)}")
    
    def load_model(self, model_path):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建embedding层（这里使用随机初始化，实际使用时应该加载原始embedding）
        from embedding_manager import EmbeddingManager
        embedding_manager = EmbeddingManager(self.config, self.word_to_idx)
        embedding_layer = embedding_manager.create_embedding_layer(trainable=False)
        
        # 创建模型
        if self.config.model == 'cnn':
            self.model = CNNTextClassifier(self.config, embedding_layer)
        elif self.config.model == 'rnn':
            self.model = RNNTextClassifier(self.config, embedding_layer)
        else:
            raise ValueError(f"不支持的模型类型: {self.config.model}")
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型已加载: {model_path}")
    
    def preprocess_text(self, text):
        """预处理文本"""
        # 转换为小写
        text = text.lower()
        
        # 去除特殊字符，保留字母、数字和空格
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
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
    
    def predict(self, text):
        """预测文本情感"""
        # 预处理文本
        processed_text = self.preprocess_text(text)
        
        # 转换为索引
        indices = self.text_to_indices(processed_text)
        
        # 转换为张量
        input_tensor = torch.LongTensor([indices]).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'text': text,
            'processed_text': processed_text,
            'predicted_class': predicted_class,
            'predicted_label': self.sentiment_labels[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy()
        }
    
    def predict_batch(self, texts):
        """批量预测"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='文本情感预测')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--vocab_path', type=str, required=True, help='词汇表文件路径')
    parser.add_argument('--text', type=str, help='要预测的文本')
    parser.add_argument('--file', type=str, help='包含多个文本的文件路径')
    
    args = parser.parse_args()
    
    # 获取配置（从模型文件中加载）
    checkpoint = torch.load(args.model_path, map_location='cpu')
    config = checkpoint['config']
    
    # 创建预测器
    predictor = TextPredictor(args.model_path, args.vocab_path, config)
    
    if args.text:
        # 预测单个文本
        result = predictor.predict(args.text)
        print(f"\n文本: {result['text']}")
        print(f"预处理后: {result['processed_text']}")
        print(f"预测情感: {result['predicted_label']} (类别: {result['predicted_class']})")
        print(f"置信度: {result['confidence']:.4f}")
        print("\n各类别概率:")
        for i, prob in enumerate(result['probabilities']):
            print(f"  {predictor.sentiment_labels[i]}: {prob:.4f}")
    
    elif args.file:
        # 批量预测
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = predictor.predict_batch(texts)
        
        print(f"\n批量预测结果 ({len(results)} 个文本):")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. 文本: {result['text']}")
            print(f"   预测: {result['predicted_label']} (置信度: {result['confidence']:.4f})")
            print()
    
    else:
        # 交互式预测
        print("交互式文本情感预测 (输入 'quit' 退出)")
        print("-" * 50)
        
        while True:
            text = input("\n请输入文本: ").strip()
            if text.lower() == 'quit':
                break
            
            if text:
                result = predictor.predict(text)
                print(f"预测情感: {result['predicted_label']} (置信度: {result['confidence']:.4f})")

if __name__ == "__main__":
    main()