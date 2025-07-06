import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools

class FeatureExtractor:
    def __init__(self, max_features=10000, min_df=2, max_df=0.95):
        """
        特征提取器
        Args:
            max_features: 最大特征数
            min_df: 最小文档频率
            max_df: 最大文档频率
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary = None
        self.word_to_idx = None
        self.idx_to_word = None
        self.tfidf_vectorizer = None  # 添加TF-IDF vectorizer属性
        self.ngram_vocabulary = None  # 新增
        self.ngram_to_idx = None      # 新增
        
    def build_vocabulary(self, texts, min_freq=2):
        """
        构建词汇表
        Args:
            texts: 文本列表（每个文本是词汇列表）
            min_freq: 最小词频
        """
        print("Building vocabulary...")
        
        # 统计词频
        word_freq = Counter()
        # 统计文档频率（包含该词的文档数）
        word_doc_freq = Counter()
        
        for text in texts:
            # 统计词频
            word_freq.update(text)
            # 统计文档频率（每个词在文档中只算一次）
            unique_words = set(text)
            word_doc_freq.update(unique_words)
        
        n_docs = len(texts)
        
        # 过滤低频词和高频词
        filtered_words = {}
        for word, freq in word_freq.items():
            doc_freq = word_doc_freq[word]
            doc_freq_ratio = doc_freq / n_docs
            
            # 检查最小词频和最大文档频率
            if freq >= min_freq and doc_freq_ratio <= self.max_df:
                filtered_words[word] = freq
        
        # 按词频排序，取前max_features个词
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        if self.max_features:
            sorted_words = sorted_words[:self.max_features]
        
        # 构建词汇表
        self.vocabulary = [word for word, freq in sorted_words]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Filtered by min_freq={min_freq}, max_df={self.max_df}")
        return self.vocabulary
    
    def extract_bow_features(self, texts):
        """
        提取Bag-of-Words特征
        Args:
            texts: 文本列表（每个文本是词汇列表）
        Returns:
            特征矩阵 (n_samples, vocab_size)
        """
        if self.vocabulary is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
        
        print("Extracting BoW features...")
        n_samples = len(texts)
        n_features = len(self.vocabulary)
        
        # 初始化特征矩阵
        features = np.zeros((n_samples, n_features), dtype=np.float32)
        
        # 计算词频
        for i, text in enumerate(texts):
            word_counts = Counter(text)
            for word, count in word_counts.items():
                if word in self.word_to_idx:
                    features[i, self.word_to_idx[word]] = count
        
        print(f"BoW features shape: {features.shape}")
        return features
    
    def extract_ngram_features(self, texts, n_range=(1, 3), fit=True):
        """
        提取N-gram特征
        Args:
            texts: 文本列表（每个文本是词汇列表）
            n_range: N-gram范围，如(1,3)表示1-gram到3-gram
            fit: 是否拟合词汇表（训练集为True，验证/测试集为False）
        Returns:
            特征矩阵
        """
        print(f"Extracting {n_range[0]}-{n_range[1]}-gram features...")
        if fit:
            ngram_freq = Counter()
            ngram_doc_freq = Counter()
            for text in texts:
                text_ngrams = set()
                for n in range(n_range[0], n_range[1] + 1):
                    ngrams = list(zip(*[text[i:] for i in range(n)]))
                    ngram_freq.update(ngrams)
                    text_ngrams.update(ngrams)
                ngram_doc_freq.update(text_ngrams)
            n_docs = len(texts)
            filtered_ngrams = {}
            for ngram, freq in ngram_freq.items():
                doc_freq = ngram_doc_freq[ngram]
                doc_freq_ratio = doc_freq / n_docs
                if freq >= self.min_df and doc_freq_ratio <= self.max_df:
                    filtered_ngrams[ngram] = freq
            sorted_ngrams = sorted(filtered_ngrams.items(), key=lambda x: x[1], reverse=True)
            if self.max_features:
                sorted_ngrams = sorted_ngrams[:self.max_features]
            self.ngram_vocabulary = [ngram for ngram, freq in sorted_ngrams]
            self.ngram_to_idx = {ngram: idx for idx, ngram in enumerate(self.ngram_vocabulary)}
        else:
            if self.ngram_to_idx is None:
                raise ValueError("N-gram vocabulary not built. Call extract_ngram_features with fit=True first.")
        n_samples = len(texts)
        n_features = len(self.ngram_to_idx)
        features = np.zeros((n_samples, n_features), dtype=np.float32)
        for i, text in enumerate(texts):
            for n in range(n_range[0], n_range[1] + 1):
                ngrams = list(zip(*[text[i:] for i in range(n)]))
                ngram_counts = Counter(ngrams)
                for ngram, count in ngram_counts.items():
                    if ngram in self.ngram_to_idx:
                        features[i, self.ngram_to_idx[ngram]] = count
        print(f"N-gram features shape: {features.shape}")
        if fit:
            print(f"Filtered by min_df={self.min_df}, max_df={self.max_df}")
        return features
    
    def extract_tfidf_features(self, texts, fit=True):
        """
        提取TF-IDF特征
        Args:
            texts: 文本列表（每个文本是词汇列表）
            fit: 是否拟合vectorizer（训练集为True，验证/测试集为False）
        Returns:
            TF-IDF特征矩阵
        """
        print("Extracting TF-IDF features...")
        
        # 将词汇列表转换为文本字符串
        text_strings = [' '.join(text) for text in texts]
        
        if fit:
            # 训练集：创建并拟合vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                lowercase=False,  # 已经预处理过了
                token_pattern=r'\S+'  # 匹配非空白字符
            )
            features = self.tfidf_vectorizer.fit_transform(text_strings).toarray()
        else:
            # 验证/测试集：使用已拟合的vectorizer
            if self.tfidf_vectorizer is None:
                raise ValueError("TF-IDF vectorizer not fitted. Call extract_tfidf_features with fit=True first.")
            features = self.tfidf_vectorizer.transform(text_strings).toarray()
        
        print(f"TF-IDF features shape: {features.shape}")
        return features
    
    def combine_features(self, feature_list, weights=None):
        """
        组合多种特征
        Args:
            feature_list: 特征矩阵列表
            weights: 权重列表
        Returns:
            组合后的特征矩阵
        """
        if weights is None:
            weights = [1.0] * len(feature_list)
        
        assert len(feature_list) == len(weights), "Feature list and weights must have same length"
        
        # 水平拼接特征
        combined_features = np.hstack([w * features for w, features in zip(weights, feature_list)])
        print(f"Combined features shape: {combined_features.shape}")
        return combined_features
    
    def normalize_features(self, features, method='l2'):
        """
        特征归一化
        Args:
            features: 特征矩阵
            method: 归一化方法 ('l2', 'l1', 'max')
        Returns:
            归一化后的特征矩阵
        """
        if method == 'l2':
            # L2归一化
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 避免除零
            return features / norms
        elif method == 'l1':
            # L1归一化
            norms = np.sum(np.abs(features), axis=1, keepdims=True)
            norms[norms == 0] = 1
            return features / norms
        elif method == 'max':
            # 最大归一化
            max_vals = np.max(features, axis=1, keepdims=True)
            max_vals[max_vals == 0] = 1
            return features / max_vals
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def get_feature_names(self):
        """
        获取特征名称
        Returns:
            特征名称列表
        """
        if self.vocabulary is None:
            return []
        return self.vocabulary.copy()

class FeatureSelector:
    def __init__(self, method='chi2', k=1000):
        """
        特征选择器
        Args:
            method: 选择方法 ('chi2', 'mutual_info', 'variance')
            k: 选择的特征数
        """
        self.method = method
        self.k = k
        self.selected_features = None
        self.selector = None
        
    def select_features(self, X, y):
        """
        特征选择
        Args:
            X: 特征矩阵
            y: 标签
        Returns:
            选择后的特征矩阵
        """
        from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, VarianceThreshold
        
        if self.method == 'chi2':
            self.selector = SelectKBest(chi2, k=self.k)
        elif self.method == 'mutual_info':
            self.selector = SelectKBest(mutual_info_classif, k=self.k)
        elif self.method == 'variance':
            self.selector = VarianceThreshold(threshold=0.01)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")
        
        X_selected = self.selector.fit_transform(X, y)
        self.selected_features = self.selector.get_support()
        
        print(f"Selected {X_selected.shape[1]} features using {self.method}")
        return X_selected
    
    def transform(self, X):
        """
        对新数据进行特征选择
        Args:
            X: 特征矩阵
        Returns:
            选择后的特征矩阵
        """
        if self.selector is None:
            raise ValueError("Feature selector not fitted. Call select_features first.")
        
        return self.selector.transform(X)

def main():
    """
    测试特征提取器
    """
    # 示例数据
    sample_texts = [
        ['this', 'is', 'a', 'good', 'movie'],
        ['this', 'movie', 'is', 'bad'],
        ['i', 'like', 'this', 'film'],
        ['i', 'hate', 'this', 'movie'],
        ['excellent', 'film', 'very', 'good']
    ]
    
    # 创建特征提取器
    extractor = FeatureExtractor(max_features=100)
    
    # 构建词汇表
    extractor.build_vocabulary(sample_texts)
    
    # 提取BoW特征
    bow_features = extractor.extract_bow_features(sample_texts)
    print("BoW features:")
    print(bow_features)
    
    # 提取N-gram特征
    ngram_features = extractor.extract_ngram_features(sample_texts, n_range=(1, 2))
    print("N-gram features:")
    print(ngram_features)
    
    # 组合特征
    combined_features = extractor.combine_features([bow_features, ngram_features])
    print("Combined features shape:", combined_features.shape)

if __name__ == "__main__":
    main() 