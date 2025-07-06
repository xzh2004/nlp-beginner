"""
预测脚本：使用训练好的LSTM+CRF模型进行命名实体识别
"""

import os
import sys
import torch
import json
import argparse
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_loader import DataManager
from model import BiLSTMCRF
from utils import create_directories, set_random_seed


def load_model(config, vocab_info, device):
    """
    加载训练好的模型
    """
    model = BiLSTMCRF(
        vocab_size=vocab_info['vocab_size'],
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_tags=vocab_info['num_labels'],
        dropout=config.dropout,
        bidirectional=config.bidirectional,
        pad_idx=vocab_info['vocab'][config.pad_token]
    )
    
    model = model.to(device)
    
    # 加载模型权重
    model_path = os.path.join(config.model_save_dir, 'best_model.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from: {model_path}")
        print(f"Best F1 score: {checkpoint['best_f1']:.4f}")
        return model, checkpoint
    else:
        raise FileNotFoundError(f"No model found at: {model_path}")


def preprocess_text(text, vocab, config):
    """
    预处理输入文本
    """
    # 分词（简单的空格分词）
    words = text.strip().split()
    
    # 转换为ID序列
    word_ids = [vocab.get(config.start_token, vocab[config.unk_token])]
    
    for word in words:
        word_id = vocab.get(word.lower(), vocab[config.unk_token])
        word_ids.append(word_id)
    
    word_ids.append(vocab.get(config.end_token, vocab[config.unk_token]))
    
    return words, word_ids


def predict_single_text(text, model, vocab, label_vocab, config, device):
    """
    对单个文本进行预测
    """
    # 预处理文本
    words, word_ids = preprocess_text(text, vocab, config)
    
    # 转换为tensor
    word_tensor = torch.tensor([word_ids], dtype=torch.long).to(device)
    lengths = torch.tensor([len(word_ids)], dtype=torch.long).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        predictions = model.predict(word_tensor, lengths)
    
    # 转换预测结果
    pred_ids = predictions[0, :lengths[0]].cpu().numpy()
    
    # 创建标签ID到标签名的映射
    id_to_label = {v: k for k, v in label_vocab.items()}
    
    # 返回结果（去掉START和END标签）
    results = []
    for i, (word, pred_id) in enumerate(zip(words, pred_ids[1:-1])):
        label = id_to_label[pred_id]
        results.append({
            'word': word,
            'label': label,
            'position': i
        })
    
    return results


def predict_batch_texts(texts, model, vocab, label_vocab, config, device):
    """
    对批量文本进行预测
    """
    results = []
    
    for text in texts:
        text_result = predict_single_text(text, model, vocab, label_vocab, config, device)
        results.append({
            'text': text,
            'entities': text_result
        })
    
    return results


def print_predictions(results):
    """
    打印预测结果
    """
    for i, result in enumerate(results):
        print(f"\nText {i+1}: {result['text']}")
        print("Entities:")
        
        current_entity = None
        entity_words = []
        
        for item in result['entities']:
            word = item['word']
            label = item['label']
            
            if label.startswith('B-'):
                # 开始新实体
                if current_entity:
                    print(f"  {current_entity}: {' '.join(entity_words)}")
                
                current_entity = label[2:]  # 去掉'B-'前缀
                entity_words = [word]
                
            elif label.startswith('I-') and current_entity and label[2:] == current_entity:
                # 继续当前实体
                entity_words.append(word)
                
            elif label == 'O':
                # 非实体
                if current_entity:
                    print(f"  {current_entity}: {' '.join(entity_words)}")
                    current_entity = None
                    entity_words = []
        
        # 处理最后一个实体
        if current_entity:
            print(f"  {current_entity}: {' '.join(entity_words)}")
        
        if not any(item['label'] != 'O' for item in result['entities']):
            print("  No entities found")


def save_predictions(results, save_path):
    """
    保存预测结果到文件
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    """
    主预测函数
    """
    parser = argparse.ArgumentParser(description='NER Prediction')
    parser.add_argument('--text', type=str, help='Single text to predict')
    parser.add_argument('--file', type=str, help='File containing texts (one per line)')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LSTM+CRF Named Entity Recognition Prediction")
    print("=" * 60)
    
    # 创建配置
    config = Config()
    
    # 创建必要的目录
    create_directories(config)
    
    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 设置随机种子
    set_random_seed(config.random_seed)
    
    # 加载数据管理器（用于获取词汇表）
    print("\nLoading vocabulary...")
    data_manager = DataManager(config)
    vocab_info = data_manager.get_vocab_info()
    
    # 加载模型
    print("\nLoading model...")
    model, checkpoint = load_model(config, vocab_info, device)
    
    # 准备文本
    texts = []
    
    if args.text:
        # 单个文本
        texts = [args.text]
        
    elif args.file:
        # 从文件读取
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
            
    elif args.interactive:
        # 交互模式
        print("\nEnter texts for prediction (press Enter twice to finish):")
        while True:
            text = input("Text: ").strip()
            if not text:
                break
            texts.append(text)
    
    else:
        # 默认示例
        example_texts = [
            "John Smith works at Microsoft in Seattle.",
            "Apple Inc. was founded by Steve Jobs in California.",
            "The Great Wall of China is a famous landmark.",
            "Barack Obama was the 44th President of the United States."
        ]
        texts = example_texts
        print("\nUsing example texts:")
        for i, text in enumerate(texts):
            print(f"{i+1}. {text}")
    
    if not texts:
        print("No texts provided for prediction.")
        return
    
    # 进行预测
    print(f"\nPredicting entities for {len(texts)} text(s)...")
    results = predict_batch_texts(texts, model, vocab_info['vocab'], 
                                 vocab_info['label_vocab'], config, device)
    
    # 打印结果
    print_predictions(results)
    
    # 保存结果
    if args.output or not args.interactive:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = args.output or f"predictions_{timestamp}.json"
        
        save_predictions(results, output_file)
        print(f"\nPredictions saved to: {output_file}")
    
    print("\nPrediction completed!")


if __name__ == "__main__":
    main() 