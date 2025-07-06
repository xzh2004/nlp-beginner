import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from config import Config
from model import ESIM
from utils import load_snli_data, build_vocab, SNLIDataset, collate_fn, load_glove_embeddings

def create_directories():
    """创建必要的目录"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

def load_and_preprocess_data(config, max_train_samples=10000, max_dev_samples=1000):
    """加载和预处理数据"""
    print("Loading data...")
    
    # 加载数据
    train_data = load_snli_data(config.train_path, max_train_samples)
    dev_data = load_snli_data(config.dev_path, max_dev_samples)
    
    print(f"Loaded {len(train_data)} training samples")
    print(f"Loaded {len(dev_data)} dev samples")
    
    # 构建词汇表
    print("Building vocabulary...")
    all_sentences = []
    for item in train_data:
        all_sentences.append(item['sentence1'])
        all_sentences.append(item['sentence2'])
    
    vocab = build_vocab(all_sentences, config.min_freq)
    print(f"Vocabulary size: {len(vocab)}")
    
    # 加载GloVe预训练embedding
    glove_embeddings = load_glove_embeddings(config.glove_path, vocab, config.embedding_dim)
    
    # 创建数据集
    train_dataset = SNLIDataset(train_data, vocab, config.max_length)
    dev_dataset = SNLIDataset(dev_data, vocab, config.max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    return train_loader, dev_loader, vocab, glove_embeddings

def train_epoch(model, train_loader, criterion, optimizer, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        sentence1 = batch['sentence1'].to(config.device)
        sentence2 = batch['sentence2'].to(config.device)
        labels = batch['labels']
        
        # 转换标签为索引
        label_indices = torch.tensor([config.label2idx[label] for label in labels]).to(config.device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(sentence1, sentence2)
        loss = criterion(outputs, label_indices)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += label_indices.size(0)
        correct += (predicted == label_indices).sum().item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100 * correct / total:.2f}%'
        })
    
    return total_loss / len(train_loader), correct / total

def evaluate(model, dev_loader, criterion, config):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Evaluating"):
            sentence1 = batch['sentence1'].to(config.device)
            sentence2 = batch['sentence2'].to(config.device)
            labels = batch['labels']
            
            # 转换标签为索引
            label_indices = torch.tensor([config.label2idx[label] for label in labels]).to(config.device)
            
            # 前向传播
            outputs = model(sentence1, sentence2)
            loss = criterion(outputs, label_indices)
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(label_indices.cpu().numpy())
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return total_loss / len(dev_loader), accuracy, all_predictions, all_labels

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
    """主训练函数"""
    config = Config()
    create_directories()
    
    print(f"Using device: {config.device}")
    
    # 加载数据
    train_loader, dev_loader, vocab, glove_embeddings = load_and_preprocess_data(config)
    
    # 保存词汇表
    torch.save(vocab, config.vocab_save_path)
    print(f"Vocabulary saved to {config.vocab_save_path}")
    
    # 创建模型（使用预训练embedding）
    model = ESIM(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_classes=config.num_classes,
        dropout=config.dropout,
        pretrained_embeddings=glove_embeddings
    ).to(config.device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 训练结果记录
    results = {
        'train_losses': [],
        'train_accuracies': [],
        'dev_losses': [],
        'dev_accuracies': [],
        'best_accuracy': 0
    }
    
    print("Starting training...")
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config)
        
        # 验证
        dev_loss, dev_acc, predictions, labels = evaluate(model, dev_loader, criterion, config)
        
        # 记录结果
        results['train_losses'].append(train_loss)
        results['train_accuracies'].append(train_acc)
        results['dev_losses'].append(dev_loss)
        results['dev_accuracies'].append(dev_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.4f}")
        
        # 保存最佳模型
        if dev_acc > results['best_accuracy']:
            results['best_accuracy'] = dev_acc
            torch.save(model.state_dict(), config.model_save_path)
            print(f"New best model saved with accuracy: {dev_acc:.4f}")
        
        # 保存当前结果
        save_results(results, os.path.join(config.results_save_path, 'training_results.json'))
    
    print(f"\nTraining completed! Best accuracy: {results['best_accuracy']:.4f}")
    
    # 最终评估
    print("\nFinal evaluation:")
    model.load_state_dict(torch.load(config.model_save_path, map_location=config.device))
    dev_loss, dev_acc, predictions, labels = evaluate(model, dev_loader, criterion, config)
    
    # 生成分类报告
    label_names = [config.idx2label[i] for i in range(config.num_classes)]
    report = classification_report(labels, predictions, target_names=label_names, output_dict=True)
    
    # 保存最终结果
    final_results = {
        'final_accuracy': dev_acc,
        'classification_report': report,
        'predictions': predictions,
        'labels': labels
    }
    
    save_results(final_results, os.path.join(config.results_save_path, 'final_results.json'))
    
    print(f"Final accuracy: {dev_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=label_names))

if __name__ == "__main__":
    main() 