import torch
import argparse
import os
import sys
import json
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import LSTMLanguageModel, GRULanguageModel
from text_generator import TextGenerator

def generate_text(model_path, prompt="", style="七言绝句", num_poems=5, 
                 max_length=100, temperature=0.8, device='auto'):
    """
    使用训练好的模型生成文本
    
    Args:
        model_path: 模型文件路径
        prompt: 起始提示
        style: 诗歌风格
        num_poems: 生成诗歌数量
        max_length: 最大生成长度
        temperature: 温度参数
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
    
    # 创建文本生成器
    generator = TextGenerator(model, vocab, device)
    
    # 生成文本
    if prompt:
        print(f"\n根据提示 '{prompt}' 生成文本:")
        generated_text = generator.generate_with_prompt(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature
        )
        print(f"生成结果: {generated_text}")
    
    # 生成诗歌
    print(f"\n生成 {num_poems} 首{style}风格的诗歌:")
    poems = generator.generate_poetry(
        style=style,
        num_poems=num_poems,
        temperature=temperature
    )
    
    for i, poem in enumerate(poems, 1):
        print(f"\n诗歌 {i}:")
        print(poem)
        print("-" * 40)
    
    return {
        'prompt': prompt,
        'style': style,
        'num_poems': num_poems,
        'temperature': temperature,
        'poems': poems,
        'model_config': model_config
    }

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='生成唐诗风格文本')
    # 添加命令行参数，指定模型文件路径
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    # 添加命令行参数，指定起始提示文本
    parser.add_argument('--prompt', type=str, default="",
                       help='起始提示文本')
    # 添加命令行参数，指定诗歌风格
    parser.add_argument('--style', type=str, default='七言绝句',
                       choices=['五言绝句', '七言绝句', '五言律诗', '七言律诗'],
                       help='诗歌风格')
    # 添加命令行参数，指定生成诗歌数量
    parser.add_argument('--num_poems', type=int, default=5,
                       help='生成诗歌数量')
    # 添加命令行参数，指定最大生成长度
    parser.add_argument('--max_length', type=int, default=100,
                       help='最大生成长度')
    # 添加命令行参数，指定温度参数
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='温度参数')
    # 添加命令行参数，指定设备
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (cpu, cuda, auto)')
    # 添加命令行参数，指定输出文件路径
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径')
    # 添加命令行参数，指定随机种子
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # 生成文本
    results = generate_text(
        model_path=args.model_path,
        prompt=args.prompt,
        style=args.style,
        num_poems=args.num_poems,
        max_length=args.max_length,
        temperature=args.temperature,
        device=args.device
    )
    
    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")

if __name__ == '__main__':
    main() 