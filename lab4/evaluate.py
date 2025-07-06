"""
评估脚本：评估训练好的LSTM+CRF模型
"""

import os
import sys
import torch
import json
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_loader import DataManager
from model import BiLSTMCRF
from trainer import NERTrainer
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


def evaluate_model(model, data_loader, device, label_names):
    """
    评估模型
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_words = []
    
    with torch.no_grad():
        for batch in data_loader:
            # 将数据移到设备上
            word_ids = batch['word_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            masks = batch['masks'].to(device)
            lengths = batch['lengths'].to(device)
            
            # 获取预测结果
            predictions = model.predict(word_ids, lengths)
            
            # 收集预测和真实标签
            for i, length in enumerate(lengths):
                pred_seq = predictions[i, :length].cpu().numpy()
                true_seq = label_ids[i, :length].cpu().numpy()
                word_seq = word_ids[i, :length].cpu().numpy()
                
                all_predictions.extend(pred_seq)
                all_labels.extend(true_seq)
                all_words.extend(word_seq)
    
    return all_predictions, all_labels, all_words


def print_detailed_results(predictions, labels, label_vocab, vocab, config):
    """
    打印详细的评估结果
    """
    # 创建标签ID到标签名的映射
    id_to_label = {v: k for k, v in label_vocab.items()}
    id_to_word = {v: k for k, v in vocab.items()}
    
    # 统计每个标签的预测情况
    label_stats = {}
    for label_id in label_vocab.values():
        label_name = id_to_label[label_id]
        label_stats[label_name] = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
    
    # 计算统计信息
    for pred, true in zip(predictions, labels):
        pred_label = id_to_label[pred]
        true_label = id_to_label[true]
        
        if pred == true:
            label_stats[pred_label]['true_positives'] += 1
        else:
            label_stats[pred_label]['false_positives'] += 1
            label_stats[true_label]['false_negatives'] += 1
    
    # 打印每个标签的详细结果
    print("\nDetailed Results by Label:")
    print("-" * 80)
    print(f"{'Label':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 80)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for label_name, stats in label_stats.items():
        tp = stats['true_positives']
        fp = stats['false_positives']
        fn = stats['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn
        
        print(f"{label_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # 计算总体指标
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
    
    print("-" * 80)
    print(f"{'Overall':<15} {total_precision:<12.4f} {total_recall:<12.4f} {total_f1:<12.4f} {total_tp + total_fn:<10}")
    print("-" * 80)


def main():
    """
    主评估函数
    """
    print("=" * 60)
    print("LSTM+CRF Named Entity Recognition Evaluation")
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
    
    # 加载数据
    print("\nLoading data...")
    data_manager = DataManager(config)
    train_loader, dev_loader, test_loader = data_manager.get_dataloaders()
    vocab_info = data_manager.get_vocab_info()
    
    # 加载模型
    print("\nLoading model...")
    model, checkpoint = load_model(config, vocab_info, device)
    
    # 创建训练器用于评估
    trainer = NERTrainer(model, config, device)
    
    # 评估验证集
    print("\nEvaluating on validation set...")
    dev_loss, dev_metrics = trainer.evaluate(dev_loader)
    
    print(f"Validation Results:")
    print(f"Loss: {dev_loss:.4f}")
    print(f"Precision: {dev_metrics['precision']:.4f}")
    print(f"Recall: {dev_metrics['recall']:.4f}")
    print(f"F1 Score: {dev_metrics['f1']:.4f}")
    
    # 评估测试集
    print("\nEvaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader)
    
    print(f"Test Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    
    # 获取详细的预测结果
    print("\nGetting detailed predictions...")
    test_predictions, test_labels, test_words = evaluate_model(
        model, test_loader, device, list(vocab_info['label_vocab'].keys())
    )
    
    # 打印详细结果
    print_detailed_results(
        test_predictions, test_labels, 
        vocab_info['label_vocab'], vocab_info['vocab'], config
    )
    
    # 保存评估结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_results = {
        'config': {
            'embedding_dim': config.embedding_dim,
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'dropout': config.dropout,
            'bidirectional': config.bidirectional,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'vocab_size': vocab_info['vocab_size'],
            'num_labels': vocab_info['num_labels']
        },
        'model_info': {
            'best_f1': checkpoint['best_f1'],
            'model_path': os.path.join(config.model_save_dir, 'best_model.pth')
        },
        'evaluation_results': {
            'validation': {
                'loss': dev_loss,
                'precision': dev_metrics['precision'],
                'recall': dev_metrics['recall'],
                'f1': dev_metrics['f1']
            },
            'test': {
                'loss': test_loss,
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1': test_metrics['f1']
            }
        },
        'timestamp': timestamp
    }
    
    results_path = os.path.join(config.results_dir, f"evaluation_results_{timestamp}.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation results saved to: {results_path}")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main() 