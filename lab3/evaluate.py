import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from config import Config
from model import ESIM
from utils import load_snli_data, SNLIDataset, collate_fn, load_vocab

def load_model_and_vocab(config):
    """加载训练好的模型和词汇表"""
    # 加载词汇表
    vocab = load_vocab(config.vocab_save_path)
    
    # 创建模型
    model = ESIM(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_classes=config.num_classes,
        dropout=config.dropout
    ).to(config.device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(config.model_save_path, map_location=config.device))
    model.eval()
    
    return model, vocab

def evaluate_model(model, test_loader, config):
    """评估模型"""
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            sentence1 = batch['sentence1'].to(config.device)
            sentence2 = batch['sentence2'].to(config.device)
            labels = batch['labels']
            
            # 转换标签为索引
            label_indices = torch.tensor([config.label2idx[label] for label in labels]).to(config.device)
            
            # 前向传播
            outputs = model(sentence1, sentence2)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(label_indices.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return all_predictions, all_labels, all_probabilities

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_class_distribution(y_true, y_pred, labels, save_path):
    """绘制类别分布"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 真实标签分布
    true_counts = [sum(1 for y in y_true if y == i) for i in range(len(labels))]
    ax1.bar(labels, true_counts, color='skyblue')
    ax1.set_title('True Label Distribution')
    ax1.set_ylabel('Count')
    
    # 预测标签分布
    pred_counts = [sum(1 for y in y_pred if y == i) for i in range(len(labels))]
    ax2.bar(labels, pred_counts, color='lightcoral')
    ax2.set_title('Predicted Label Distribution')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_errors(predictions, labels, test_data, config, save_path):
    """分析错误预测"""
    errors = []
    
    for i, (pred, true) in enumerate(zip(predictions, labels)):
        if pred != true:
            pred_label = config.idx2label[pred]
            true_label = config.idx2label[true]
            
            errors.append({
                'index': i,
                'sentence1': test_data[i]['sentence1'],
                'sentence2': test_data[i]['sentence2'],
                'true_label': true_label,
                'predicted_label': pred_label
            })
    
    # 保存错误分析
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    
    return errors

def save_results(results, file_path):
    """保存结果"""
    # 转换numpy类型为Python原生类型
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results = convert_numpy(results)
    
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    """主评估函数"""
    config = Config()
    
    print("Loading model and vocabulary...")
    try:
        model, vocab = load_model_and_vocab(config)
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first using train.py")
        return
    
    print("Loading test data...")
    test_data = load_snli_data(config.test_path, max_samples=1000)  # 限制测试样本数量
    
    # 创建测试数据集
    test_dataset = SNLIDataset(test_data, vocab, config.max_length)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print("Evaluating model...")
    predictions, labels, probabilities = evaluate_model(model, test_loader, config)
    
    # 计算准确率
    accuracy = accuracy_score(labels, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # 生成分类报告
    label_names = [config.idx2label[i] for i in range(config.num_classes)]
    report = classification_report(labels, predictions, target_names=label_names, output_dict=True)
    
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=label_names))
    
    # 保存结果
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'predictions': predictions,
        'labels': labels,
        'probabilities': probabilities
    }
    
    results_path = os.path.join(config.results_save_path, 'test_results.json')
    save_results(results, results_path)
    
    # 绘制混淆矩阵
    cm_path = os.path.join(config.results_save_path, 'confusion_matrix.png')
    plot_confusion_matrix(labels, predictions, label_names, cm_path)
    
    # 绘制类别分布
    dist_path = os.path.join(config.results_save_path, 'class_distribution.png')
    plot_class_distribution(labels, predictions, label_names, dist_path)
    
    # 分析错误预测
    errors_path = os.path.join(config.results_save_path, 'error_analysis.json')
    errors = analyze_errors(predictions, labels, test_data, config, errors_path)
    
    print(f"\nResults saved to {config.results_save_path}")
    print(f"Number of errors: {len(errors)}")
    
    # 显示一些错误示例
    if errors:
        print("\nError examples:")
        for i, error in enumerate(errors[:5]):  # 显示前5个错误
            print(f"\nError {i+1}:")
            print(f"Sentence 1: {error['sentence1']}")
            print(f"Sentence 2: {error['sentence2']}")
            print(f"True: {error['true_label']}, Predicted: {error['predicted_label']}")

if __name__ == "__main__":
    main() 