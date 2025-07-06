import torch
import torch.nn as nn
import argparse
import os
import sys
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from model import LSTMLanguageModel, GRULanguageModel, calculate_perplexity
from trainer import LanguageModelTrainer
from text_generator import TextGenerator

def main():
    parser = argparse.ArgumentParser(description='训练字符级语言模型')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='../poetryFromTang.txt',
                       help='数据文件路径')
    parser.add_argument('--seq_length', type=int, default=50,
                       help='序列长度')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--min_freq', type=int, default=1,
                       help='最小字符频率')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'gru'],
                       help='模型类型: lstm 或 gru')
    parser.add_argument('--embedding_dim', type=int, default=64,
                       help='嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='层数')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout率')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                       help='权重衰减')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                       help='梯度裁剪')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='models',
                       help='模型保存目录')
    parser.add_argument('--vocab_path', type=str, default=None,
                       help='词汇表保存路径')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (cpu, cuda, auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    train_loader, val_loader, test_loader, vocab = load_data(
        filepath=args.data_path,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        min_freq=args.min_freq,
        vocab_path=args.vocab_path
    )
    
    # 创建模型
    print(f"创建{args.model_type.upper()}模型...")
    if args.model_type == 'lstm':
        model = LSTMLanguageModel(
            vocab_size=vocab.vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:  # gru
        model = GRULanguageModel(
            vocab_size=vocab.vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数数量: {total_params:,}")
    
    # 创建训练器
    trainer = LanguageModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad
    )
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{args.model_type}_{timestamp}")
    
    # 训练模型
    print(f"开始训练，模型将保存到: {save_dir}")
    trainer.train(
        num_epochs=args.num_epochs,
        save_dir=save_dir,
        save_every=10
    )
    
    # 计算测试集困惑度
    print("\n计算测试集困惑度...")
    test_perplexity, test_loss = calculate_perplexity(model, test_loader, device, vocab)
    print(f"测试集困惑度: {test_perplexity:.2f}")
    print(f"测试集损失: {test_loss:.4f}")
    
    # 保存最终结果
    results = {
        'model_type': args.model_type,
        'vocab_size': vocab.vocab_size,
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'seq_length': args.seq_length,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'total_params': total_params,
        'test_perplexity': test_perplexity,
        'test_loss': test_loss,
        'best_val_loss': min(trainer.val_losses),
        'best_val_perplexity': min(trainer.val_perplexities),
        'timestamp': timestamp
    }
    
    import json
    with open(os.path.join(save_dir, 'final_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n训练完成！结果已保存到: {save_dir}")
    print(f"最终测试困惑度: {test_perplexity:.2f}")

def train_lstm():
    """训练LSTM模型"""
    print("=== 训练LSTM语言模型 ===")
    main()

def train_gru():
    """训练GRU模型"""
    print("=== 训练GRU语言模型 ===")
    # 修改默认参数
    import sys
    sys.argv.extend(['--model_type', 'gru'])
    main()

if __name__ == '__main__':
    main() 