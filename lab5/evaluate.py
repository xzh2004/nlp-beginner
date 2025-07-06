import torch
import argparse
import os
import sys
import json
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from model import LSTMLanguageModel, GRULanguageModel, calculate_perplexity
from text_generator import TextGenerator

def evaluate_model(model_path, data_path='../poetryFromTang.txt', 
                  seq_length=50, batch_size=32, device='auto'):
    """
    评估训练好的模型
    
    Args:
        model_path: 模型文件路径
        data_path: 数据文件路径
        seq_length: 序列长度
        batch_size: 批次大小
        device: 设备
    """
    
    # 设置设备
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取模型配置
    model_config = checkpoint['model_config']
    vocab = checkpoint['vocab']
    
    print(f"模型配置: {model_config}")
    print(f"词汇表大小: {vocab.vocab_size}")
    
    # 创建模型
    if 'lstm' in model_path.lower():
        model = LSTMLanguageModel(**model_config)
    else:
        model = GRULanguageModel(**model_config)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 加载数据
    print("加载数据...")
    _, _, test_loader, _ = load_data(
        filepath=data_path,
        seq_length=seq_length,
        batch_size=batch_size,
        vocab_path=None  # 使用模型中的词汇表
    )
    
    # 计算困惑度
    print("计算困惑度...")
    perplexity, loss = calculate_perplexity(model, test_loader, device, vocab)
    
    print(f"测试集困惑度: {perplexity:.2f}")
    print(f"测试集损失: {loss:.4f}")
    
    # 生成示例文本
    print("\n生成示例文本...")
    generator = TextGenerator(model, vocab, device)
    
    # 生成几首诗歌
    poems = generator.generate_poetry(style="七言绝句", num_poems=3, temperature=0.8)
    
    print("生成的诗歌:")
    for i, poem in enumerate(poems, 1):
        print(f"\n诗歌 {i}:")
        print(poem)
    
    # 根据提示生成文本
    prompts = ["春", "月", "山", "水"]
    print("\n根据提示生成文本:")
    for prompt in prompts:
        generated = generator.generate_with_prompt(prompt, max_length=30, temperature=0.8)
        print(f"提示 '{prompt}': {generated}")
    
    return {
        'perplexity': perplexity,
        'loss': loss,
        'model_config': model_config,
        'vocab_size': vocab.vocab_size,
        'poems': poems
    }

def main():
    parser = argparse.ArgumentParser(description='评估语言模型')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--data_path', type=str, default='../poetryFromTang.txt',
                       help='数据文件路径')
    parser.add_argument('--seq_length', type=int, default=50,
                       help='序列长度')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (cpu, cuda, auto)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出结果文件路径')
    
    args = parser.parse_args()
    
    # 评估模型
    results = evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")

if __name__ == '__main__':
    main() 