#!/usr/bin/env python3
"""
分析数据集，检查类别分布和其他统计信息
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader

def analyze_dataset():
    """
    分析数据集的详细统计信息
    """
    print("=" * 80)
    print("数据集详细分析")
    print("=" * 80)
    
    # 创建数据加载器
    loader = DataLoader()
    
    # 加载原始数据
    train_data, test_data = loader.load_data()
    
    print("\n1. 原始数据基本信息:")
    print("-" * 40)
    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")
    print(f"训练集列名: {list(train_data.columns)}")
    print(f"测试集列名: {list(test_data.columns)}")
    
    # 分析类别分布
    print("\n2. 类别分布分析:")
    print("-" * 40)
    
    sentiment_counts = train_data['Sentiment'].value_counts().sort_index()
    total_samples = len(train_data)
    
    print("训练集类别分布:")
    for sentiment, count in sentiment_counts.items():
        percentage = count / total_samples * 100
        print(f"  类别 {sentiment}: {count:6d} 个样本 ({percentage:5.1f}%)")
    
    # 检查类别不平衡
    max_count = sentiment_counts.max()
    min_count = sentiment_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\n类别不平衡分析:")
    print(f"  最多类别样本数: {max_count}")
    print(f"  最少类别样本数: {min_count}")
    print(f"  不平衡比例: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 3:
        print("  ⚠️  警告: 数据集存在严重的类别不平衡问题!")
    elif imbalance_ratio > 2:
        print("  ⚠️  注意: 数据集存在一定的类别不平衡")
    else:
        print("  ✅ 数据集类别分布相对平衡")
    
    # 分析文本长度分布
    print("\n3. 文本长度分析:")
    print("-" * 40)
    
    text_lengths = train_data['Phrase'].str.len()
    print(f"平均文本长度: {text_lengths.mean():.1f} 字符")
    print(f"文本长度标准差: {text_lengths.std():.1f} 字符")
    print(f"最短文本: {text_lengths.min()} 字符")
    print(f"最长文本: {text_lengths.max()} 字符")
    print(f"中位数长度: {text_lengths.median():.1f} 字符")
    
    # 按类别分析文本长度
    print(f"\n各类别平均文本长度:")
    for sentiment in sorted(train_data['Sentiment'].unique()):
        class_texts = train_data[train_data['Sentiment'] == sentiment]['Phrase']
        avg_length = class_texts.str.len().mean()
        print(f"  类别 {sentiment}: {avg_length:.1f} 字符")
    
    # 预处理数据
    print("\n4. 预处理后数据分析:")
    print("-" * 40)
    
    processed_train = loader.preprocess_dataset(train_data)
    
    # 分析词汇分布
    all_tokens = []
    for tokens in processed_train['tokens']:
        all_tokens.extend(tokens)
    
    token_counts = Counter(all_tokens)
    unique_tokens = len(token_counts)
    total_tokens = len(all_tokens)
    
    print(f"总词汇数量: {total_tokens:,}")
    print(f"唯一词汇数量: {unique_tokens:,}")
    print(f"平均每样本词汇数: {total_tokens / len(processed_train):.1f}")
    
    # 显示最常见的词汇
    print(f"\n最常见的20个词汇:")
    for token, count in token_counts.most_common(20):
        print(f"  '{token}': {count:6d} 次")
    
    # 分析稀有词汇
    rare_tokens = [token for token, count in token_counts.items() if count == 1]
    print(f"\n只出现1次的词汇数量: {len(rare_tokens)} ({len(rare_tokens)/unique_tokens*100:.1f}%)")
    
    # 绘制类别分布图
    print("\n5. 生成可视化图表:")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 类别分布柱状图
    axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values, color='skyblue')
    axes[0, 0].set_title('Class Distribution')
    axes[0, 0].set_xlabel('Sentiment Class')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 文本长度分布直方图
    axes[0, 1].hist(text_lengths, bins=50, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Text Length Distribution')
    axes[0, 1].set_xlabel('Text Length (Characters)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 各类别文本长度箱线图
    sentiment_labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    class_lengths = []
    for sentiment in sorted(train_data['Sentiment'].unique()):
        class_texts = train_data[train_data['Sentiment'] == sentiment]['Phrase']
        class_lengths.append(class_texts.str.len().values)
    
    axes[1, 0].boxplot(class_lengths, labels=sentiment_labels)
    axes[1, 0].set_title('Text Length Distribution by Class')
    axes[1, 0].set_xlabel('Sentiment Class')
    axes[1, 0].set_ylabel('Text Length (Characters)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 词汇频率分布（前100个）
    top_100_tokens = dict(token_counts.most_common(100))
    axes[1, 1].plot(range(1, 101), list(top_100_tokens.values()), color='orange')
    axes[1, 1].set_title('Word Frequency Distribution (Top 100)')
    axes[1, 1].set_xlabel('Word Rank')
    axes[1, 1].set_ylabel('Occurrence Count')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
    print("数据分布图表已保存为 'data_analysis.png'")
    
    # 分析结果总结
    print("\n6. 分析结果总结:")
    print("-" * 40)
    
    print("主要发现:")
    print(f"1. 数据集包含 {total_samples:,} 个训练样本")
    print(f"2. 共有 {len(sentiment_counts)} 个情感类别")
    print(f"3. 类别不平衡比例: {imbalance_ratio:.2f}:1")
    print(f"4. 平均文本长度: {text_lengths.mean():.1f} 字符")
    print(f"5. 词汇表大小: {unique_tokens:,} 个唯一词汇")
    
    if imbalance_ratio > 3:
        print("\n⚠️  建议:")
        print("- 考虑使用类别权重或重采样技术")
        print("- 使用更适合不平衡数据的评估指标（如F1-score）")
        print("- 考虑使用数据增强技术")
    
    print("\n✅ 分析完成!")

if __name__ == "__main__":
    analyze_dataset()