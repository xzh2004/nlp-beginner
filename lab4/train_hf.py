"""
使用Hugging Face数据集的训练脚本：训练LSTM+CRF模型进行命名实体识别
"""

import os
import sys
import torch
import time
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from hf_data_loader import HFDataManager
from model import BiLSTMCRF
from trainer import NERTrainer
from utils import create_directories, set_random_seed


def main():
    """
    主训练函数
    """
    print("=" * 60)
    print("LSTM+CRF Named Entity Recognition Training (Hugging Face)")
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
    print("\nLoading data from Hugging Face...")
    data_manager = HFDataManager(config)
    train_loader, dev_loader, test_loader = data_manager.get_dataloaders()
    vocab_info = data_manager.get_vocab_info()
    
    print(f"Vocabulary size: {vocab_info['vocab_size']}")
    print(f"Number of labels: {vocab_info['num_labels']}")
    print(f"Labels: {list(vocab_info['label_vocab'].keys())}")
    
    # 创建模型
    print("\nCreating model...")
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
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 创建训练器
    trainer = NERTrainer(model, config, device)
    
    # 开始训练
    print(f"\nStarting training for {config.num_epochs} epochs...")
    start_time = time.time()
    
    training_results = trainer.train(train_loader, dev_loader, config.num_epochs)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存训练曲线
    curves_path = os.path.join(config.results_dir, f"hf_training_curves_{timestamp}.png")
    trainer.save_training_curves(curves_path)
    print(f"Training curves saved to: {curves_path}")
    
    # 保存训练结果
    results_path = os.path.join(config.results_dir, f"hf_training_results_{timestamp}.json")
    final_results = {
        'config': {
            'embedding_dim': config.embedding_dim,
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'dropout': config.dropout,
            'bidirectional': config.bidirectional,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'num_epochs': config.num_epochs,
            'vocab_size': vocab_info['vocab_size'],
            'num_labels': vocab_info['num_labels'],
            'data_source': 'hugging_face_conll2003'
        },
        'training_results': training_results,
        'best_f1': trainer.best_f1,
        'training_time': training_results['training_time'],
        'timestamp': timestamp
    }
    trainer.save_results(final_results, results_path)
    print(f"Training results saved to: {results_path}")
    
    # 最终评估
    print("\nFinal evaluation on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader)
    
    print(f"\nFinal Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Best Dev F1: {trainer.best_f1:.4f}")
    
    # 保存最终结果
    final_results['test_results'] = {
        'test_loss': test_loss,
        'test_metrics': test_metrics
    }
    trainer.save_results(final_results, results_path)
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/60:.2f} minutes")
    print("Training completed successfully!")


if __name__ == "__main__":
    main() 