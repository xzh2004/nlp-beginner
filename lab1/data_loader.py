import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re

class DataLoader:
    def __init__(self, train_file='data/train.tsv', test_file='data/test.tsv'):
        """
        数据加载器
        Args:
            train_file: 训练数据文件路径
            test_file: 测试数据文件路径
        """
        self.train_file = train_file
        self.test_file = test_file
        self.train_data = None
        self.test_data = None
        
    def load_data(self):
        """
        加载训练和测试数据
        """
        print("Loading training data...")
        self.train_data = pd.read_csv(self.train_file, sep='\t')
        print(f"Training data shape: {self.train_data.shape}")
        
        print("Loading test data...")
        self.test_data = pd.read_csv(self.test_file, sep='\t')
        print(f"Test data shape: {self.test_data.shape}")
        
        return self.train_data, self.test_data
    
    def preprocess_text(self, text):
        """
        文本预处理
        Args:
            text: 输入文本
        Returns:
            预处理后的文本
        """
        if pd.isna(text):
            return ""
        
        # 转换为小写
        text = text.lower()
        
        # 去除特殊字符，保留字母、数字和空格
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        分词（按空格分割）
        Args:
            text: 预处理后的文本
        Returns:
            词汇列表
        """
        if not text:
            return []
        return text.split()
    
    def preprocess_dataset(self, data, text_column='Phrase'):
        """
        预处理整个数据集
        Args:
            data: 数据集
            text_column: 文本列名
        Returns:
            预处理后的数据集
        """
        print("Preprocessing dataset...")
        
        # 复制数据避免修改原始数据
        processed_data = data.copy()
        
        # 预处理文本
        processed_data['processed_text'] = processed_data[text_column].apply(self.preprocess_text)
        
        # 分词
        processed_data['tokens'] = processed_data['processed_text'].apply(self.tokenize)
        
        # 过滤空文本
        processed_data = processed_data[processed_data['processed_text'].str.len() > 0]
        
        print(f"Processed data shape: {processed_data.shape}")
        
        # 输出几行处理过的数据
        self.display_processed_samples(processed_data, text_column)
        
        return processed_data
    
    def display_processed_samples(self, data, text_column='Phrase', num_samples=5):
        """
        显示处理过的数据样本
        Args:
            data: 处理后的数据集
            text_column: 原始文本列名
            num_samples: 显示的样本数量
        """
        print(f"\n{'='*60}")
        print(f"显示 {num_samples} 个处理过的数据样本:")
        print(f"{'='*60}")
        
        # 随机选择样本
        sample_data = data.sample(n=min(num_samples, len(data)), random_state=42)
        
        for i, (_, row) in enumerate(sample_data.iterrows(), 1):
            print(f"\n样本 {i}:")
            print(f"  原始文本: {row[text_column]}")
            print(f"  预处理后: {row['processed_text']}")
            print(f"  分词结果: {row['tokens']}")
            if 'Sentiment' in row:
                print(f"  情感标签: {row['Sentiment']}")
            print("-" * 40)
    
    def split_dataset(self, data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        划分数据集为训练集、验证集、测试集
        Args:
            data: 数据集
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_state: 随机种子
        Returns:
            train_data, val_data, test_data
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        # 首先划分出测试集
        train_val_data, test_data = train_test_split(
            data, test_size=test_ratio, random_state=random_state, stratify=data['Sentiment']
        )
        
        # 从剩余数据中划分训练集和验证集
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_data, val_data = train_test_split(
            train_val_data, test_size=val_ratio_adjusted, random_state=random_state, 
            stratify=train_val_data['Sentiment']
        )
        
        print(f"Train set: {train_data.shape}")
        print(f"Validation set: {val_data.shape}")
        print(f"Test set: {test_data.shape}")
        
        return train_data, val_data, test_data
    
    def get_class_distribution(self, data):
        """
        获取类别分布
        Args:
            data: 数据集
        Returns:
            类别分布字典
        """
        return data['Sentiment'].value_counts().sort_index().to_dict()
    
    def print_data_info(self):
        """
        打印数据信息
        """
        if self.train_data is not None:
            print("\n=== Training Data Info ===")
            print(f"Shape: {self.train_data.shape}")
            print("Columns:", list(self.train_data.columns))
            print("Class distribution:")
            print(self.get_class_distribution(self.train_data))
            
            # 显示一些样本
            print("\nSample data:")
            print(self.train_data[['Phrase', 'Sentiment']].head())
        
        if self.test_data is not None:
            print("\n=== Test Data Info ===")
            print(f"Shape: {self.test_data.shape}")
            print("Columns:", list(self.test_data.columns))

def main():
    """
    测试数据加载器
    """
    print("=" * 60)
    print("数据加载器测试")
    print("=" * 60)
    
    # 创建数据加载器
    loader = DataLoader()
    
    # 加载数据
    train_data, test_data = loader.load_data()
    
    # 打印数据信息
    loader.print_data_info()
    
    # 预处理训练数据
    processed_train = loader.preprocess_dataset(train_data)
    
    # 显示更多统计信息
    print(f"\n{'='*60}")
    print("数据预处理统计信息:")
    print(f"{'='*60}")
    
    # 计算平均文本长度
    avg_original_length = processed_train['Phrase'].str.len().mean()
    avg_processed_length = processed_train['processed_text'].str.len().mean()
    avg_token_count = processed_train['tokens'].str.len().mean()
    
    print(f"平均原始文本长度: {avg_original_length:.1f} 字符")
    print(f"平均预处理后长度: {avg_processed_length:.1f} 字符")
    print(f"平均词汇数量: {avg_token_count:.1f} 个")
    
    # 显示词汇分布
    all_tokens = []
    for tokens in processed_train['tokens']:
        all_tokens.extend(tokens)
    
    unique_tokens = set(all_tokens)
    print(f"总词汇数量: {len(all_tokens)}")
    print(f"唯一词汇数量: {len(unique_tokens)}")
    
    # 显示最常见的词汇
    from collections import Counter
    token_counts = Counter(all_tokens)
    print(f"\n最常见的10个词汇:")
    for token, count in token_counts.most_common(10):
        print(f"  '{token}': {count} 次")
    
    # 划分数据集
    train_split, val_split, test_split = loader.split_dataset(processed_train)
    
    print(f"\n{'='*60}")
    print("数据集划分信息:")
    print(f"{'='*60}")
    print("训练集类别分布:", loader.get_class_distribution(train_split))
    print("验证集类别分布:", loader.get_class_distribution(val_split))
    print("测试集类别分布:", loader.get_class_distribution(test_split))

if __name__ == "__main__":
    main() 