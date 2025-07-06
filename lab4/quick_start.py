"""
快速开始脚本：LSTM+CRF命名实体识别
"""

import os
import sys
import torch
import time
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_loader import DataManager
from model import BiLSTMCRF
from trainer import NERTrainer
from utils import create_directories, set_random_seed


def check_data_files(config):
    """
    检查数据文件是否存在
    """
    required_files = [
        f"{config.data_dir}/{config.train_file}",
        f"{config.data_dir}/{config.dev_file}",
        f"{config.data_dir}/{config.test_file}"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing data files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease download the CONLL 2003 dataset and place the files in the data/ directory.")
        print("You can download it from: https://www.clips.uantwerpen.be/conll2003/ner/")
        return False
    
    return True


def demo_training():
    """
    演示训练过程
    """
    print("=" * 60)
    print("LSTM+CRF NER Training Demo")
    print("=" * 60)
    
    # 创建配置
    config = Config()
    
    # 检查数据文件
    if not check_data_files(config):
        return
    
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
    print(f"Total parameters: {total_params:,}")
    
    # 创建训练器
    trainer = NERTrainer(model, config, device)
    
    # 开始训练（使用较少的epoch进行演示）
    demo_epochs = min(5, config.num_epochs)
    print(f"\nStarting demo training for {demo_epochs} epochs...")
    
    start_time = time.time()
    training_results = trainer.train(train_loader, dev_loader, demo_epochs)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存训练曲线
    curves_path = os.path.join(config.results_dir, f"demo_training_curves_{timestamp}.png")
    trainer.save_training_curves(curves_path)
    print(f"Training curves saved to: {curves_path}")
    
    # 保存训练结果
    results_path = os.path.join(config.results_dir, f"demo_training_results_{timestamp}.json")
    final_results = {
        'config': {
            'embedding_dim': config.embedding_dim,
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'dropout': config.dropout,
            'bidirectional': config.bidirectional,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'demo_epochs': demo_epochs,
            'vocab_size': vocab_info['vocab_size'],
            'num_labels': vocab_info['num_labels']
        },
        'training_results': training_results,
        'best_f1': trainer.best_f1,
        'training_time': training_results['training_time'],
        'timestamp': timestamp
    }
    trainer.save_results(final_results, results_path)
    print(f"Training results saved to: {results_path}")
    
    total_time = time.time() - start_time
    print(f"\nDemo training completed in {total_time/60:.2f} minutes")
    print(f"Best F1 score: {trainer.best_f1:.4f}")


def demo_prediction():
    """
    演示预测过程
    """
    print("=" * 60)
    print("LSTM+CRF NER Prediction Demo")
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
    
    # 检查是否有训练好的模型
    model_path = os.path.join(config.model_save_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print("No trained model found. Please run training first.")
        return
    
    # 加载数据管理器（用于获取词汇表）
    print("\nLoading vocabulary...")
    data_manager = DataManager(config)
    vocab_info = data_manager.get_vocab_info()
    
    # 加载模型
    print("\nLoading model...")
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
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model with F1: {checkpoint['best_f1']:.4f}")
    
    # 示例文本
    example_texts = [
        "John Smith works at Microsoft in Seattle.",
        "Apple Inc. was founded by Steve Jobs in California.",
        "The Great Wall of China is a famous landmark.",
        "Barack Obama was the 44th President of the United States.",
        "The University of Oxford is located in England."
    ]
    
    print("\nPredicting entities for example texts:")
    for i, text in enumerate(example_texts):
        print(f"{i+1}. {text}")
    
    # 进行预测
    print(f"\nPredicting entities...")
    
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(example_texts):
            print(f"\nText {i+1}: {text}")
            print("Entities:")
            
            # 预处理文本
            words = text.strip().split()
            word_ids = [vocab_info['vocab'][config.start_token]]
            
            for word in words:
                word_id = vocab_info['vocab'].get(word.lower(), vocab_info['vocab'][config.unk_token])
                word_ids.append(word_id)
            
            word_ids.append(vocab_info['vocab'][config.end_token])
            
            # 转换为tensor
            word_tensor = torch.tensor([word_ids], dtype=torch.long).to(device)
            lengths = torch.tensor([len(word_ids)], dtype=torch.long).to(device)
            
            # 预测
            predictions = model.predict(word_tensor, lengths)
            pred_ids = predictions[0, :lengths[0]].cpu().numpy()
            
            # 创建标签ID到标签名的映射
            id_to_label = {v: k for k, v in vocab_info['label_vocab'].items()}
            
            # 打印结果
            current_entity = None
            entity_words = []
            
            for j, (word, pred_id) in enumerate(zip(words, pred_ids[1:-1])):
                label = id_to_label[pred_id]
                
                if label.startswith('B-'):
                    # 开始新实体
                    if current_entity:
                        print(f"  {current_entity}: {' '.join(entity_words)}")
                    
                    current_entity = label[2:]  # 去掉'B-'前缀
                    entity_words = [word]
                    
                elif label.startswith('I-') and current_entity and label[2:] == current_entity:
                    # 继续当前实体
                    entity_words.append(word)
                    
                elif label == 'O':
                    # 非实体
                    if current_entity:
                        print(f"  {current_entity}: {' '.join(entity_words)}")
                        current_entity = None
                        entity_words = []
            
            # 处理最后一个实体
            if current_entity:
                print(f"  {current_entity}: {' '.join(entity_words)}")
            
            if not any(id_to_label[pred_ids[j+1]] != 'O' for j in range(len(words))):
                print("  No entities found")
    
    print("\nPrediction demo completed!")


def main():
    """
    主函数
    """
    print("LSTM+CRF Named Entity Recognition Quick Start")
    print("=" * 60)
    print("1. Demo Training")
    print("2. Demo Prediction")
    print("3. Exit")
    
    while True:
        choice = input("\nPlease select an option (1-3): ").strip()
        
        if choice == '1':
            demo_training()
            break
        elif choice == '2':
            demo_prediction()
            break
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main() 