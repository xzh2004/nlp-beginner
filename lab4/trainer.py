"""
训练器：实现模型训练、验证和早停功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import os
from tqdm import tqdm
from utils import compute_metrics, plot_training_curves, save_results, set_random_seed


class NERTrainer:
    """
    NER模型训练器
    """
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # 训练历史
        self.train_losses = []
        self.dev_losses = []
        self.train_f1s = []
        self.dev_f1s = []
        self.best_f1 = 0.0
        self.patience_counter = 0
        
        # 设置随机种子
        set_random_seed(config.random_seed)
    
    def train_epoch(self, train_loader, epoch):
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移到设备上
            word_ids = batch['word_ids'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            masks = batch['masks'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            loss = self.model(word_ids, lengths, label_ids, masks)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            # 参数更新
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            
            # 获取预测结果用于计算指标
            with torch.no_grad():
                predictions = self.model.predict(word_ids, lengths)
                
                # 收集预测和真实标签
                for i, length in enumerate(lengths):
                    pred_seq = predictions[i, :length].cpu().numpy()
                    true_seq = label_ids[i, :length].cpu().numpy()
                    
                    all_predictions.extend(pred_seq)
                    all_labels.extend(true_seq)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        # 计算平均损失和F1分数
        avg_loss = total_loss / len(train_loader)
        metrics = compute_metrics(all_labels, all_predictions, list(range(self.model.num_tags)))
        
        return avg_loss, metrics['f1']
    
    def evaluate(self, dev_loader):
        """
        在验证集上评估模型
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc='Evaluating'):
                # 将数据移到设备上
                word_ids = batch['word_ids'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)
                masks = batch['masks'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                
                # 前向传播
                loss = self.model(word_ids, lengths, label_ids, masks)
                total_loss += loss.item()
                
                # 获取预测结果
                predictions = self.model.predict(word_ids, lengths)
                
                # 收集预测和真实标签
                for i, length in enumerate(lengths):
                    pred_seq = predictions[i, :length].cpu().numpy()
                    true_seq = label_ids[i, :length].cpu().numpy()
                    
                    all_predictions.extend(pred_seq)
                    all_labels.extend(true_seq)
        
        # 计算平均损失和指标
        avg_loss = total_loss / len(dev_loader)
        metrics = compute_metrics(all_labels, all_predictions, list(range(self.model.num_tags)))
        
        return avg_loss, metrics
    
    def train(self, train_loader, dev_loader, num_epochs):
        """
        训练模型
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_f1 = self.train_epoch(train_loader, epoch)
            
            # 验证
            dev_loss, dev_metrics = self.evaluate(dev_loader)
            dev_f1 = dev_metrics['f1']
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.dev_losses.append(dev_loss)
            self.train_f1s.append(train_f1)
            self.dev_f1s.append(dev_f1)
            
            # 学习率调度
            self.scheduler.step(dev_f1)
            
            # 打印结果
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
            print(f"  Dev Loss: {dev_loss:.4f}, Dev F1: {dev_f1:.4f}")
            print(f"  Dev Precision: {dev_metrics['precision']:.4f}")
            print(f"  Dev Recall: {dev_metrics['recall']:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if dev_f1 > self.best_f1:
                self.best_f1 = dev_f1
                self.patience_counter = 0
                self.save_best_model()
                print(f"  New best model saved! F1: {dev_f1:.4f}")
            else:
                self.patience_counter += 1
                print(f"  No improvement for {self.patience_counter} epochs")
            
            # 早停
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            print("-" * 50)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time/60:.2f} minutes")
        print(f"Best F1 score: {self.best_f1:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'dev_losses': self.dev_losses,
            'train_f1s': self.train_f1s,
            'dev_f1s': self.dev_f1s,
            'best_f1': self.best_f1,
            'training_time': training_time
        }
    
    def save_best_model(self):
        """
        保存最佳模型
        """
        os.makedirs(self.config.model_save_dir, exist_ok=True)
        
        model_path = os.path.join(self.config.model_save_dir, 'best_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_f1': self.best_f1,
            'train_losses': self.train_losses,
            'dev_losses': self.dev_losses,
            'train_f1s': self.train_f1s,
            'dev_f1s': self.dev_f1s
        }, model_path)
    
    def save_training_curves(self, save_path):
        """
        保存训练曲线
        """
        plot_training_curves(
            self.train_losses, self.dev_losses,
            self.train_f1s, self.dev_f1s,
            save_path
        )
    
    def save_results(self, results, save_path):
        """
        保存训练结果
        """
        save_results(results, save_path)
    
    def load_best_model(self):
        """
        加载最佳模型
        """
        model_path = os.path.join(self.config.model_save_dir, 'best_model.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_f1 = checkpoint['best_f1']
            self.train_losses = checkpoint['train_losses']
            self.dev_losses = checkpoint['dev_losses']
            self.train_f1s = checkpoint['train_f1s']
            self.dev_f1s = checkpoint['dev_f1s']
            print(f"Loaded best model with F1: {self.best_f1:.4f}")
            return True
        else:
            print("No saved model found")
            return False 