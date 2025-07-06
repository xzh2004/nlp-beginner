import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import time
from datetime import datetime

class LanguageModelTrainer:
    """语言模型训练器"""
    
    def __init__(self, model, train_loader, val_loader, vocab, device, 
                 learning_rate=0.001, weight_decay=1e-5, clip_grad=1.0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.device = device
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.8,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.char2idx.get('<PAD>', -1)
        )
        
        # 梯度裁剪
        self.clip_grad = clip_grad
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
        self.learning_rates = []
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(self.device), y.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            output, _ = self.model(x)
            
            # 计算损失
            loss = self.criterion(
                output.view(-1, self.vocab.vocab_size), 
                y.view(-1)
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            
            # 更新参数
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * x.size(0) * x.size(1)
            total_tokens += x.size(0) * x.size(1)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / total_tokens:.4f}'
            })
        
        avg_loss = total_loss / total_tokens
        return avg_loss
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # 前向传播
                output, _ = self.model(x)
                
                # 计算损失
                loss = self.criterion(
                    output.view(-1, self.vocab.vocab_size), 
                    y.view(-1)
                )
                
                # 统计
                total_loss += loss.item() * x.size(0) * x.size(1)
                total_tokens += x.size(0) * x.size(1)
        
        avg_loss = total_loss / total_tokens
        return avg_loss
    
    def train(self, num_epochs, save_dir='models', save_every=5, patience=5):
        """训练模型，添加早停机制"""
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 计算困惑度
            train_perplexity = np.exp(train_loss)
            val_perplexity = np.exp(val_loss)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_perplexities.append(train_perplexity)
            self.val_perplexities.append(val_perplexity)
            self.learning_rates.append(current_lr)
            
            # 计算时间
            epoch_time = time.time() - epoch_start_time
            
            # 打印结果
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
            print(f"  Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.2f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(os.path.join(save_dir, 'best_model.pth'))
                print(f"  保存最佳模型 (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  验证损失未改善，耐心计数: {patience_counter}/{patience}")
                
                # 早停
                if patience_counter >= patience:
                    print(f"  早停！验证损失连续{patience}个epoch未改善")
                    break
            
            # 定期保存模型
            if (epoch + 1) % save_every == 0:
                self.save_model(os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总用时: {total_time:.2f}s")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        print(f"最佳验证困惑度: {np.exp(best_val_loss):.2f}")
        
        # 保存训练历史
        self.save_training_history(save_dir)
        
        # 绘制训练曲线
        self.plot_training_curves(save_dir)
    
    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'model_config': {
                'vocab_size': self.vocab.vocab_size,
                'embedding_dim': self.model.embedding_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers
            }
        }, filepath)
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
    
    def save_training_history(self, save_dir):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_perplexities': self.train_perplexities,
            'val_perplexities': self.val_perplexities,
            'learning_rates': self.learning_rates
        }
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    def plot_training_curves(self, save_dir):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 困惑度曲线
        axes[0, 1].plot(self.train_perplexities, label='Train Perplexity')
        axes[0, 1].plot(self.val_perplexities, label='Val Perplexity')
        axes[0, 1].set_title('Training and Validation Perplexity')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率曲线
        axes[1, 0].plot(self.learning_rates)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
        
        # 验证损失和困惑度对比
        axes[1, 1].plot(self.val_losses, label='Val Loss', color='blue')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss', color='blue')
        axes[1, 1].tick_params(axis='y', labelcolor='blue')
        
        ax2 = axes[1, 1].twinx()
        ax2.plot(self.val_perplexities, label='Val Perplexity', color='red')
        ax2.set_ylabel('Perplexity', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        axes[1, 1].set_title('Validation Loss vs Perplexity')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(save_dir, f'training_curves_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show() 