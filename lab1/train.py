import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import DataLoader
from feature_extraction import FeatureExtractor, FeatureSelector
from model import LogisticRegression, SoftmaxRegression

class TextClassifier:
    """
    文本分类器主类
    """
    def __init__(self, config):
        """
        初始化文本分类器
        Args:
            config: 配置字典
        """
        self.config = config
        self.data_loader = None
        self.feature_extractor = None
        self.feature_selector = None
        self.model = None
        self.results = {}
        
    def load_and_preprocess_data(self):
        """
        加载和预处理数据
        """
        print("=" * 50)
        print("Loading and preprocessing data...")
        
        # 创建数据加载器
        self.data_loader = DataLoader(
            train_file=self.config['data']['train_file'],
            test_file=self.config['data']['test_file']
        )
        
        # 加载数据
        train_data, test_data = self.data_loader.load_data()
        
        # 预处理数据
        processed_train = self.data_loader.preprocess_dataset(train_data)
        processed_test = self.data_loader.preprocess_dataset(test_data)
        
        # 划分数据集
        train_split, val_split, test_split = self.data_loader.split_dataset(
            processed_train,
            train_ratio=self.config['data']['train_ratio'],
            val_ratio=self.config['data']['val_ratio'],
            test_ratio=self.config['data']['test_ratio']
        )
        
        return train_split, val_split, test_split, processed_test
    
    def extract_features(self, train_data, val_data, test_data):
        """
        提取特征
        """
        print("=" * 50)
        print("Extracting features...")
        
        # 创建特征提取器
        self.feature_extractor = FeatureExtractor(
            max_features=self.config['features']['max_features'],
            min_df=self.config['features']['min_df'],
            max_df=self.config['features']['max_df']
        )
        
        # 获取所有文本用于构建词汇表
        all_texts = (train_data['tokens'].tolist() + 
                    val_data['tokens'].tolist() + 
                    test_data['tokens'].tolist())
        
        # 构建词汇表
        self.feature_extractor.build_vocabulary(all_texts, min_freq=self.config['features']['min_freq'])
        
        # 提取特征
        feature_type = self.config['features']['type']
        
        if feature_type == 'bow':
            X_train = self.feature_extractor.extract_bow_features(train_data['tokens'].tolist())
            X_val = self.feature_extractor.extract_bow_features(val_data['tokens'].tolist())
            X_test = self.feature_extractor.extract_bow_features(test_data['tokens'].tolist())
            
        elif feature_type == 'n-gram':
            n_range = self.config['features']['ngram_range']
            # 对于N-gram特征，需要在训练集上拟合，然后在验证/测试集上转换
            X_train = self.feature_extractor.extract_ngram_features(train_data['tokens'].tolist(), n_range, fit=True)
            X_val = self.feature_extractor.extract_ngram_features(val_data['tokens'].tolist(), n_range, fit=False)
            X_test = self.feature_extractor.extract_ngram_features(test_data['tokens'].tolist(), n_range, fit=False)
            
        elif feature_type == 'tfidf':
            X_train = self.feature_extractor.extract_tfidf_features(train_data['tokens'].tolist(), fit=True)
            X_val = self.feature_extractor.extract_tfidf_features(val_data['tokens'].tolist(), fit=False)
            X_test = self.feature_extractor.extract_tfidf_features(test_data['tokens'].tolist(), fit=False)
            
        elif feature_type == 'combined':
            # 组合多种特征
            bow_train = self.feature_extractor.extract_bow_features(train_data['tokens'].tolist())
            bow_val = self.feature_extractor.extract_bow_features(val_data['tokens'].tolist())
            bow_test = self.feature_extractor.extract_bow_features(test_data['tokens'].tolist())
            
            ngram_train = self.feature_extractor.extract_ngram_features(train_data['tokens'].tolist(), (1, 2))
            ngram_val = self.feature_extractor.extract_ngram_features(val_data['tokens'].tolist(), (1, 2))
            ngram_test = self.feature_extractor.extract_ngram_features(test_data['tokens'].tolist(), (1, 2))
            
            X_train = self.feature_extractor.combine_features([bow_train, ngram_train], [1.0, 0.5])
            X_val = self.feature_extractor.combine_features([bow_val, ngram_val], [1.0, 0.5])
            X_test = self.feature_extractor.combine_features([bow_test, ngram_test], [1.0, 0.5])
        
        # 特征归一化
        if self.config['features']['normalize']:
            normalize_method = self.config['features']['normalize_method']
            X_train = self.feature_extractor.normalize_features(X_train, normalize_method)
            X_val = self.feature_extractor.normalize_features(X_val, normalize_method)
            X_test = self.feature_extractor.normalize_features(X_test, normalize_method)
        
        # 特征选择
        if self.config['features']['feature_selection']['enabled']:
            self.feature_selector = FeatureSelector(
                method=self.config['features']['feature_selection']['method'],
                k=self.config['features']['feature_selection']['k']
            )
            X_train = self.feature_selector.select_features(X_train, train_data['Sentiment'].values)
            X_val = self.feature_selector.transform(X_val)
            X_test = self.feature_selector.transform(X_test)
        
        # 获取标签
        y_train = train_data['Sentiment'].values
        y_val = val_data['Sentiment'].values
        y_test = test_data['Sentiment'].values
        
        print(f"Feature shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train, X_val, y_train, y_val):
        """
        训练模型
        """
        print("=" * 50)
        print("Training model...")
        
        # 确定类别数
        n_classes = len(np.unique(y_train))
        print(f"Number of classes: {n_classes}")
        
        # 选择模型
        model_type = self.config['model']['type']
        
        # 获取损失函数类型（如果配置中存在）
        loss_function = self.config['model'].get('loss_function', 'cross_entropy')
        print(f"Using loss function: {loss_function}")
        
        if model_type == 'logistic':
            # 对于多分类问题，将标签转换为二分类（one-vs-rest）
            if n_classes > 2:
                print("Using one-vs-rest approach for multi-class classification")
                self.model = []
                for i in range(n_classes):
                    print(f"Training classifier for class {i}")
                    y_binary = (y_train == i).astype(int)
                    y_val_binary = (y_val == i).astype(int)
                    
                    model = LogisticRegression(
                        learning_rate=self.config['model']['learning_rate'],
                        max_iter=self.config['model']['max_iter'],
                        batch_size=self.config['model']['batch_size'],
                        regularization=self.config['model']['regularization'],
                        lambda_reg=self.config['model']['lambda_reg'],
                        early_stopping=self.config['model']['early_stopping'],
                        patience=self.config['model']['patience'],
                        plot_training=self.config['model'].get('plot_training', True),
                        config=self.config,
                        shuffle=self.config['model'].get('shuffle', True),
                        batch_type=self.config['model'].get('batch_type', 'mini_batch'),
                        loss_function=loss_function
                    )
                    model.fit(X_train, y_binary, X_val, y_val_binary)
                    self.model.append(model)
            else:
                self.model = LogisticRegression(
                    learning_rate=self.config['model']['learning_rate'],
                    max_iter=self.config['model']['max_iter'],
                    batch_size=self.config['model']['batch_size'],
                    regularization=self.config['model']['regularization'],
                    lambda_reg=self.config['model']['lambda_reg'],
                    early_stopping=self.config['model']['early_stopping'],
                    patience=self.config['model']['patience'],
                    plot_training=self.config['model'].get('plot_training', True),
                    config=self.config,
                    shuffle=self.config['model'].get('shuffle', True),
                    batch_type=self.config['model'].get('batch_type', 'mini_batch'),
                    loss_function=loss_function
                )
                self.model.fit(X_train, y_train, X_val, y_val)
        
        elif model_type == 'softmax':
            self.model = SoftmaxRegression(
                n_classes=n_classes,
                learning_rate=self.config['model']['learning_rate'],
                max_iter=self.config['model']['max_iter'],
                batch_size=self.config['model']['batch_size'],
                regularization=self.config['model']['regularization'],
                lambda_reg=self.config['model']['lambda_reg'],
                early_stopping=self.config['model']['early_stopping'],
                patience=self.config['model']['patience'],
                plot_training=self.config['model'].get('plot_training', True),
                config=self.config,
                shuffle=self.config['model'].get('shuffle', True),
                batch_type=self.config['model'].get('batch_type', 'mini_batch'),
                loss_function=loss_function
            )
            self.model.fit(X_train, y_train, X_val, y_val)
    
    def predict(self, X):
        """
        预测
        """
        if isinstance(self.model, list):
            # one-vs-rest方法
            predictions = []
            for model in self.model:
                pred = model.predict_proba(X)
                predictions.append(pred)
            predictions = np.array(predictions).T
            return np.argmax(predictions, axis=1)
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        预测概率
        """
        if isinstance(self.model, list):
            # one-vs-rest方法
            predictions = []
            for model in self.model:
                pred = model.predict_proba(X)
                predictions.append(pred)
            predictions = np.array(predictions).T
            return predictions
        else:
            return self.model.predict_proba(X)
    
    def predict_test_file(self, test_file_path, output_file=None):
        """
        预测test.tsv文件
        """
        print(f"\n开始预测测试文件: {test_file_path}")
        
        # 检查测试文件是否存在
        if not os.path.exists(test_file_path):
            print(f"警告: 测试文件不存在: {test_file_path}")
            return None
        
        # 读取测试文件
        try:
            df = pd.read_csv(test_file_path, sep='\t', encoding='utf-8')
        except:
            try:
                df = pd.read_csv(test_file_path, sep=',', encoding='utf-8')
            except:
                df = pd.read_csv(test_file_path, encoding='utf-8')
        
        print(f"测试文件包含 {len(df)} 行数据")
        print(f"列名: {list(df.columns)}")
        
        # 找到文本列
        text_column = None
        for col in df.columns:
            if 'text' in col.lower() or col.lower() in ['text', 'sentence', 'content', 'review', 'phrase']:
                text_column = col
                break
        
        if text_column is None:
            # 如果没有找到明确的文本列，使用第一列
            text_column = df.columns[0]
        
        # 特殊处理：如果第一列是PhraseId（数字），则使用Phrase列
        if text_column == 'PhraseId' and 'Phrase' in df.columns:
            text_column = 'Phrase'
        
        print(f"使用列 '{text_column}' 作为文本列")
        
        # 情感标签映射
        sentiment_labels = {
            0: "非常负面",
            1: "负面", 
            2: "中性",
            3: "正面",
            4: "非常正面"
        }
        
        # 预处理测试数据
        processed_test = self.data_loader.preprocess_dataset(df, text_column)
        
        # 提取特征
        feature_type = self.config['features']['type']
        
        if feature_type == 'bow':
            X_test = self.feature_extractor.extract_bow_features(processed_test['tokens'].tolist())
        elif feature_type == 'n-gram':
            n_range = self.config['features']['ngram_range']
            X_test = self.feature_extractor.extract_ngram_features(processed_test['tokens'].tolist(), n_range, fit=False)
        elif feature_type == 'tfidf':
            X_test = self.feature_extractor.extract_tfidf_features(processed_test['tokens'].tolist(), fit=False)
        elif feature_type == 'combined':
            bow_test = self.feature_extractor.extract_bow_features(processed_test['tokens'].tolist())
            ngram_test = self.feature_extractor.extract_ngram_features(processed_test['tokens'].tolist(), (1, 2))
            X_test = self.feature_extractor.combine_features([bow_test, ngram_test], [1.0, 0.5])
        
        # 特征归一化
        if self.config['features']['normalize']:
            normalize_method = self.config['features']['normalize_method']
            X_test = self.feature_extractor.normalize_features(X_test, normalize_method)
        
        # 特征选择
        if self.config['features']['feature_selection']['enabled']:
            if self.feature_selector is None:
                raise ValueError("Feature selector not available. Make sure to train the model first.")
            X_test = self.feature_selector.transform(X_test)  # 使用已保存的选择器
        
        # 预测
        print("开始预测...")
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        # 计算置信度
        if probabilities.ndim == 2:
            confidences = np.max(probabilities, axis=1)
        else:
            confidences = np.abs(probabilities - 0.5) * 2  # 对于二分类
        
        print(f"预测完成！共处理 {len(df)} 行数据")
        
        # 创建结果DataFrame
        result_df = df.copy()
        
        # 处理长度不匹配的问题
        if len(predictions) != len(result_df):
            print(f"警告: 预测结果长度({len(predictions)})与原始数据长度({len(result_df)})不匹配")
            print("这通常是因为预处理时过滤了空文本")
            
            # 创建一个新的DataFrame，只包含有预测结果的行
            valid_indices = processed_test.index
            result_df = result_df.loc[valid_indices].copy()
        
        result_df['predicted_class'] = predictions
        result_df['predicted_label'] = [sentiment_labels[p] for p in predictions]
        result_df['confidence'] = confidences
        
        # 添加各类别概率
        if probabilities.ndim == 2:
            for i in range(probabilities.shape[1]):
                result_df[f'prob_class_{i}'] = probabilities[:, i]
        else:
            # 二分类情况
            result_df['prob_class_0'] = 1 - probabilities
            result_df['prob_class_1'] = probabilities
        
        # 统计预测结果
        print("\n预测结果统计:")
        print("-" * 50)
        for i in range(len(sentiment_labels)):
            count = (predictions == i).sum()
            percentage = count / len(predictions) * 100
            print(f"{sentiment_labels[i]}: {count} 个 ({percentage:.2f}%)")
        
        # 保存结果
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"test_predictions_{timestamp}.csv"
        
        result_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n预测结果已保存到: {output_file}")
        
        return result_df
    
    def plot_beautiful_training_curves(self, save_path=None):
        """
        绘制美观的训练曲线
        """
        if not hasattr(self.model, 'training_history') or not self.model.training_history:
            print("No training history available for plotting.")
            return
        
        # 设置中文字体和样式
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('seaborn-v0_8')
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Beautiful Training Curves', fontsize=20, fontweight='bold', y=0.95)
        
        # 获取训练历史
        history = self.model.training_history
        epochs = [h['epoch'] for h in history]
        train_losses = [h['train_loss'] for h in history]
        
        # 1. 训练损失曲线
        axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2.5, alpha=0.8, label='Training Loss')
        axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 训练vs验证损失
        if 'val_loss' in history[0]:
            val_losses = [h['val_loss'] for h in history]
            axes[0, 1].plot(epochs, train_losses, 'b-', linewidth=2.5, alpha=0.8, label='Training Loss')
            axes[0, 1].plot(epochs, val_losses, 'r-', linewidth=2.5, alpha=0.8, label='Validation Loss')
            axes[0, 1].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('Loss', fontsize=12)
            axes[0, 1].legend(fontsize=11)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 验证准确率
        if 'val_acc' in history[0]:
            val_accs = [h['val_acc'] for h in history]
            axes[1, 0].plot(epochs, val_accs, 'g-', linewidth=2.5, alpha=0.8, label='Validation Accuracy')
            axes[1, 0].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('Accuracy', fontsize=12)
            axes[1, 0].legend(fontsize=11)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 损失和准确率对比
        if 'val_loss' in history[0] and 'val_acc' in history[0]:
            ax2 = axes[1, 1].twinx()
            line1 = axes[1, 1].plot(epochs, val_losses, 'r-', linewidth=2.5, alpha=0.8, label='Validation Loss')
            line2 = ax2.plot(epochs, val_accs, 'g-', linewidth=2.5, alpha=0.8, label='Validation Accuracy')
            axes[1, 1].set_title('Loss vs Accuracy', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Loss', fontsize=12, color='red')
            ax2.set_ylabel('Accuracy', fontsize=12, color='green')
            
            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            axes[1, 1].legend(lines, labels, fontsize=11)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"training_plots/beautiful_training_curves_{timestamp}.png"
        
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"Beautiful training curves saved to: {save_path}")
        
        return save_path
    
    def evaluate(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        评估模型
        """
        print("=" * 50)
        print("Evaluating model...")
        
        # 训练集评估
        y_train_pred = self.predict(X_train)
        train_results = self._compute_metrics(y_train, y_train_pred, 'train')
        
        # 验证集评估
        y_val_pred = self.predict(X_val)
        val_results = self._compute_metrics(y_val, y_val_pred, 'validation')
        
        # 测试集评估
        y_test_pred = self.predict(X_test)
        test_results = self._compute_metrics(y_test, y_test_pred, 'test')
        
        # 保存结果
        self.results = {
            'train': train_results,
            'validation': val_results,
            'test': test_results,
            'config': self.config
        }
        
        # 打印结果
        print("\nResults Summary:")
        print(f"Train Accuracy: {train_results['accuracy']:.4f}")
        print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        
        return self.results
    
    def _compute_metrics(self, y_true, y_pred, split_name):
        """
        计算评估指标
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        print(f"\n{split_name.capitalize()} Set Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred.tolist()
        }
    
    def save_results(self, output_dir='results'):
        """
        保存结果
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f'results_{timestamp}.json')
        
        # 转换numpy数组为列表以便JSON序列化
        results_copy = {}
        for split, metrics in self.results.items():
            if split != 'config':
                results_copy[split] = {k: v for k, v in metrics.items()}
        
        results_copy['config'] = self.config
        
        with open(results_file, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"Results saved to {results_file}")
        return results_file

def get_default_config():
    """
    获取默认配置
    """
    return {
        'data': {
            'train_file': 'data/train.tsv',
            'test_file': 'data/test.tsv',
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15
        },
        'features': {
            'type': 'bow',  # 'bow', 'ngram', 'tfidf', 'combined'
            'max_features': 5000,
            'min_df': 2,
            'max_df': 0.95,
            'min_freq': 2,
            'ngram_range': (1, 2),
            'normalize': True,
            'normalize_method': 'l2',
            'feature_selection': {
                'enabled': False,
                'method': 'chi2',
                'k': 1000
            }
        },
        'model': {
            'type': 'logistic',  # 'logistic', 'softmax'
            'loss_function': 'cross_entropy',  # 'sigmoid', 'relu', 'hinge', 'zero_one', 'cross_entropy'
            'learning_rate': 0.01,
            'max_iter': 1000,
            'batch_size': 32,
            'batch_type': 'mini_batch',  # 'mini_batch', 'batch', 'stochastic'
            'shuffle': True,  # 是否打乱数据
            'regularization': 'l2',
            'lambda_reg': 0.01,
            'early_stopping': True,
            'patience': 10,
            'plot_training': True
        }
    }

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Text Classification with Logistic/Softmax Regression')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--feature_type', type=str, default='bow', 
                       choices=['bow', 'ngram', 'tfidf', 'combined'],
                       help='Feature extraction type')
    parser.add_argument('--model_type', type=str, default='logistic',
                       choices=['logistic', 'softmax'],
                       help='Model type')
    parser.add_argument('--loss_function', type=str, default='relu',
                       choices=['sigmoid', 'relu', 'hinge', 'zero_one', 'cross_entropy'],
                       help='Loss function type')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--max_iter', type=int, default=1000,
                       help='Maximum iterations')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--batch_type', type=str, default='mini_batch',
                       choices=['mini_batch', 'batch', 'stochastic'],
                       help='Batch type for gradient descent')
    parser.add_argument('--shuffle', action='store_true', default=True,
                       help='Shuffle data during training')
    parser.add_argument('--no_shuffle', action='store_true',
                       help='Disable data shuffling during training')
    
    args = parser.parse_args()
    
    # 获取配置
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
        config['features']['type'] = args.feature_type
        config['model']['type'] = args.model_type
        config['model']['loss_function'] = args.loss_function
        config['model']['learning_rate'] = args.learning_rate
        config['model']['max_iter'] = args.max_iter
        config['model']['batch_size'] = args.batch_size
        config['model']['batch_type'] = args.batch_type
        # 处理shuffle参数
        if args.no_shuffle:
            config['model']['shuffle'] = False
        else:
            config['model']['shuffle'] = args.shuffle
    
    print("Configuration:")
    print(json.dumps(config, indent=2))
    
    # 创建分类器
    classifier = TextClassifier(config)
    
    # 加载和预处理数据
    train_data, val_data, test_data, test_processed = classifier.load_and_preprocess_data()
    
    # 提取特征
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.extract_features(
        train_data, val_data, test_data
    )
    
    # 训练模型
    classifier.train_model(X_train, X_val, y_train, y_val)
    
    # 评估模型
    results = classifier.evaluate(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # 绘制美观的训练曲线
    classifier.plot_beautiful_training_curves()
    
    # 预测test.tsv文件
    test_file_path = 'data/test.tsv'
    if os.path.exists(test_file_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_output = f"test_predictions_{timestamp}.csv"
        classifier.predict_test_file(test_file_path, prediction_output)
    else:
        print(f"测试文件不存在: {test_file_path}")
    
    # 保存结果
    classifier.save_results()

if __name__ == "__main__":
    main() 