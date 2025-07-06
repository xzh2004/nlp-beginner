import argparse
import os

def get_config():
    """获取配置参数"""
    parser = argparse.ArgumentParser(description='Deep Learning Text Classification')
    
    # 模型选择
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'rnn'],
                       help='选择模型类型: cnn 或 rnn')
    
    # Word Embedding 初始化方式
    parser.add_argument('--embedding', type=str, default='random', 
                       choices=['random', 'glove', 'word2vec'],
                       help='Word embedding初始化方式: random, glove, word2vec')
    
    # GloVe 维度选择
    parser.add_argument('--glove_dim', type=int, default=100, choices=[50, 100, 200, 300],
                       help='GloVe embedding维度 (当使用glove时)')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='../lab1/data',
                       help='数据目录路径')
    parser.add_argument('--glove_dir', type=str, default='glove.6B',
                       help='GloVe数据目录路径')
    parser.add_argument('--max_seq_length', type=int, default=50,
                       help='最大序列长度')
    parser.add_argument('--min_word_freq', type=int, default=2,
                       help='最小词频')
    
    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=100,
                       help='Embedding维度')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='类别数量')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout率')
    parser.add_argument('--adaptive_dropout', action='store_true',
                       help='根据embedding维度自适应调整dropout率')
    
    # CNN 特定参数
    parser.add_argument('--num_filters', type=int, default=100,
                       help='CNN卷积核数量')
    parser.add_argument('--filter_sizes', type=str, default='3,4,5',
                       help='CNN卷积核大小，用逗号分隔')
    
    # RNN 特定参数
    parser.add_argument('--num_layers', type=int, default=2,
                       help='RNN层数')
    parser.add_argument('--bidirectional', action='store_true',
                       help='是否使用双向RNN')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='测试集比例')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    
    # 其他参数
    parser.add_argument('--save_model', action='store_true',
                       help='是否保存模型')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='模型保存目录')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='日志打印间隔')
    
    args = parser.parse_args()
    
    # 处理filter_sizes参数
    args.filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
    
    # 创建模型保存目录
    if args.save_model:
        os.makedirs(args.model_dir, exist_ok=True)
    
    return args

def print_config(args):
    """打印配置信息"""
    print("=" * 60)
    print("配置信息:")
    print("=" * 60)
    print(f"模型类型: {args.model}")
    print(f"Word Embedding: {args.embedding}")
    if args.embedding == 'glove':
        print(f"GloVe维度: {args.glove_dim}")
    print(f"最大序列长度: {args.max_seq_length}")
    print(f"Embedding维度: {args.embedding_dim}")
    print(f"隐藏层维度: {args.hidden_dim}")
    print(f"类别数量: {args.num_classes}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"Dropout率: {args.dropout}")
    
    if args.model == 'cnn':
        print(f"卷积核数量: {args.num_filters}")
        print(f"卷积核大小: {args.filter_sizes}")
    elif args.model == 'rnn':
        print(f"RNN层数: {args.num_layers}")
        print(f"双向RNN: {args.bidirectional}")
    
    print("=" * 60)