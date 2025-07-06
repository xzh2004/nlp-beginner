import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

class Trainer:
    """模型训练器"""
    
    def __init__(self, config, model, train_loader, val_loader, test_loader):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 将模型移到设备上
        self.model.to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # 最佳模型
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        
        # 早停机制
        self.patience = 5
        self.counter = 0
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 获取数据
            texts = batch['text'].to(self.device)
            labels = batch['label'].squeeze().to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(texts)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 打印进度
            if batch_idx % self.config.log_interval == 0:
                print(f'Batch [{batch_idx}/{len(self.train_loader)}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Accuracy: {100 * correct / total:.2f}%')
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_accuracy = 100 * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate(self, data_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 获取数据
                texts = batch['text'].to(self.device)
                labels = batch['label'].squeeze().to(self.device)
                
                # 前向传播
                outputs = self.model(texts)
                loss = self.criterion(outputs, labels)
                
                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 保存预测结果
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = total_loss / len(data_loader)
        epoch_accuracy = 100 * correct / total
        
        return epoch_loss, epoch_accuracy, all_predictions, all_labels
    
    def train(self):
        """训练模型"""
        print("开始训练...")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print(f"测试集大小: {len(self.test_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch [{epoch+1}/{self.config.num_epochs}]")
            print("-" * 50)
            
            # 训练
            train_loss, train_accuracy = self.train_epoch()
            
            # 验证
            val_loss, val_accuracy, _, _ = self.validate(self.val_loader)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 保存训练历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            # 打印结果
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%")
            
            # 保存最佳模型
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_model_state = self.model.state_dict().copy()
                self.counter = 0  # 重置计数器
                print(f"新的最佳验证准确率: {val_accuracy:.2f}%")
            else:
                self.counter += 1
                print(f"验证准确率未提升，计数器: {self.counter}/{self.patience}")
            
            # 早停检查
            if self.counter >= self.patience:
                print(f"早停触发！{self.patience} 个epoch验证准确率未提升")
                break
        
        # 训练完成
        training_time = time.time() - start_time
        print(f"\n训练完成，耗时: {training_time:.2f}秒")
        print(f"最佳验证准确率: {self.best_val_accuracy:.2f}%")
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
    
    def test(self):
        """测试模型"""
        print("\n开始测试...")
        
        # 测试
        test_loss, test_accuracy, predictions, labels = self.validate(self.test_loader)
        
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试准确率: {test_accuracy:.2f}%")
        
        # 详细评估
        print("\n分类报告:")
        print(classification_report(labels, predictions))
        
        # 混淆矩阵
        cm = confusion_matrix(labels, predictions)
        print("\n混淆矩阵:")
        print(cm)
        
        return test_accuracy, predictions, labels
    
    def save_model(self, filepath):
        """保存模型"""
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': self.best_val_accuracy
        }, filepath)
        
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        
        print(f"模型已从 {filepath} 加载")
    
    def plot_training_curves(self, save_path=None):
        """Draw beautiful training curves (English version, avoid Chinese font issues)"""
        # Use only English fonts
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('#f8f9fa')
        
        # Color scheme
        colors = {
            'train_loss': '#2E86AB',      # blue
            'val_loss': '#A23B72',        # purple
            'train_acc': '#F18F01',       # orange
            'val_acc': '#C73E1D',         # red
            'grid': '#E9ECEF',            # light gray
            'background': '#FFFFFF'       # white
        }
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, 
                color=colors['train_loss'], 
                linewidth=2.5, 
                marker='o', 
                markersize=6, 
                markeredgecolor='white',
                markeredgewidth=1.5,
                label='Train Loss',
                alpha=0.9)
        
        ax1.plot(epochs, self.val_losses, 
                color=colors['val_loss'], 
                linewidth=2.5, 
                marker='s', 
                markersize=6, 
                markeredgecolor='white',
                markeredgewidth=1.5,
                label='Validation Loss',
                alpha=0.9)
        
        ax1.set_facecolor(colors['background'])
        ax1.grid(True, alpha=0.3, color=colors['grid'])
        ax1.set_title('Train & Validation Loss', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
        ax1.set_xlabel('Epoch', fontsize=12, color='#34495E')
        ax1.set_ylabel('Loss', fontsize=12, color='#34495E')
        ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
        ax1.tick_params(colors='#34495E')
        
        # Value annotation
        if len(self.train_losses) > 0:
            ax1.annotate(f'Final Train Loss: {self.train_losses[-1]:.4f}', 
                        xy=(len(self.train_losses), self.train_losses[-1]),
                        xytext=(len(self.train_losses)*0.7, max(self.train_losses)*0.8),
                        arrowprops=dict(arrowstyle='->', color=colors['train_loss'], alpha=0.7),
                        fontsize=10, color=colors['train_loss'])
            
            ax1.annotate(f'Final Val Loss: {self.val_losses[-1]:.4f}', 
                        xy=(len(self.val_losses), self.val_losses[-1]),
                        xytext=(len(self.val_losses)*0.7, max(self.val_losses)*0.6),
                        arrowprops=dict(arrowstyle='->', color=colors['val_loss'], alpha=0.7),
                        fontsize=10, color=colors['val_loss'])
        
        # Accuracy curves
        ax2.plot(epochs, self.train_accuracies, 
                color=colors['train_acc'], 
                linewidth=2.5, 
                marker='o', 
                markersize=6, 
                markeredgecolor='white',
                markeredgewidth=1.5,
                label='Train Accuracy',
                alpha=0.9)
        
        ax2.plot(epochs, self.val_accuracies, 
                color=colors['val_acc'], 
                linewidth=2.5, 
                marker='s', 
                markersize=6, 
                markeredgecolor='white',
                markeredgewidth=1.5,
                label='Validation Accuracy',
                alpha=0.9)
        
        ax2.set_facecolor(colors['background'])
        ax2.grid(True, alpha=0.3, color=colors['grid'])
        ax2.set_title('Train & Validation Accuracy', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
        ax2.set_xlabel('Epoch', fontsize=12, color='#34495E')
        ax2.set_ylabel('Accuracy (%)', fontsize=12, color='#34495E')
        ax2.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=11)
        ax2.tick_params(colors='#34495E')
        
        # Value annotation
        if len(self.train_accuracies) > 0:
            ax2.annotate(f'Final Train Acc: {self.train_accuracies[-1]:.2f}%', 
                        xy=(len(self.train_accuracies), self.train_accuracies[-1]),
                        xytext=(len(self.train_accuracies)*0.7, min(self.train_accuracies)*0.8),
                        arrowprops=dict(arrowstyle='->', color=colors['train_acc'], alpha=0.7),
                        fontsize=10, color=colors['train_acc'])
            
            ax2.annotate(f'Final Val Acc: {self.val_accuracies[-1]:.2f}%', 
                        xy=(len(self.val_accuracies), self.val_accuracies[-1]),
                        xytext=(len(self.val_accuracies)*0.7, min(self.val_accuracies)*0.6),
                        arrowprops=dict(arrowstyle='->', color=colors['val_acc'], alpha=0.7),
                        fontsize=10, color=colors['val_acc'])
        
        # Model info
        model_info = f"Model: {self.config.model.upper()} | Embedding: {self.config.embedding.upper()} | Best Val Acc: {self.best_val_accuracy:.2f}%"
        fig.suptitle(model_info, fontsize=14, color='#2C3E50', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
            print(f"Beautiful training curves saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, filepath):
        """保存训练结果"""
        results = {
            'config': vars(self.config),
            'best_val_accuracy': self.best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"训练结果已保存到: {filepath}")