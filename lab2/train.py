#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config, print_config
from data_loader import DataProcessor
from embedding_manager import EmbeddingManager
from cnn_model import CNNTextClassifier, CNNTextClassifierWithBatchNorm
from rnn_model import RNNTextClassifier, GRUTextClassifier, AttentionRNNTextClassifier
from trainer import Trainer

def set_random_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_model(config, embedding_layer):
    """根据配置创建模型"""
    if config.model == 'cnn':
        print("创建CNN模型...")
        model = CNNTextClassifier(config, embedding_layer)
    elif config.model == 'rnn':
        print("创建RNN模型...")
        model = RNNTextClassifier(config, embedding_layer)
    else:
        raise ValueError(f"不支持的模型类型: {config.model}")
    
    return model

def predict_test_file(model, data_processor, config, test_file_path, output_file):
    """预测test.tsv文件"""
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
    
    # 设置模型为评估模式
    model.eval()
    device = next(model.parameters()).device
    
    # 预测
    predictions = []
    confidences = []
    all_probabilities = []
    
    print("开始预测...")
    with torch.no_grad():
        for i, row in df.iterrows():
            if i % 100 == 0:
                print(f"已处理 {i}/{len(df)} 行...")
            
            text = row[text_column]
            
            # 预处理文本
            processed_text = data_processor.preprocess_text(text)
            
            # 转换为索引
            indices = data_processor.text_to_indices(processed_text)
            input_tensor = torch.LongTensor([indices]).to(device)
            
            # 预测
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            predictions.append(predicted_class)
            confidences.append(confidence)
            all_probabilities.append(probabilities[0].cpu().numpy())
    
    print(f"预测完成！共处理 {len(df)} 行数据")
    
    # 创建结果DataFrame
    result_df = df.copy()
    result_df['predicted_class'] = predictions
    result_df['predicted_label'] = [sentiment_labels[p] for p in predictions]
    result_df['confidence'] = confidences
    
    # 添加各类别概率
    for i in range(5):
        result_df[f'prob_class_{i}'] = [prob[i] for prob in all_probabilities]
    
    # 统计预测结果
    print("\n预测结果统计:")
    print("-" * 50)
    for i in range(5):
        count = predictions.count(i)
        percentage = count / len(predictions) * 100
        print(f"{sentiment_labels[i]}: {count} 个 ({percentage:.2f}%)")
    
    # 保存结果
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n预测结果已保存到: {output_file}")
    
    return result_df

def main():
    """主函数"""
    # 获取配置
    config = get_config()
    
    # 自适应dropout调整
    if config.adaptive_dropout and config.embedding == 'glove':
        # 根据GloVe维度调整dropout
        if config.glove_dim == 50:
            config.dropout = 0.3
        elif config.glove_dim == 100:
            config.dropout = 0.5
        elif config.glove_dim == 200:
            config.dropout = 0.6
        elif config.glove_dim == 300:
            config.dropout = 0.7
        print(f"自适应调整dropout率为: {config.dropout}")
    
    # 设置随机种子
    set_random_seed(config.random_seed)
    
    # 打印配置信息
    print_config(config)
    
    # 创建数据处理器
    print("\n初始化数据处理器...")
    data_processor = DataProcessor(config)
    
    # 加载数据
    train_data, test_data = data_processor.load_data()
    
    # 处理数据
    train_loader, val_loader, test_loader = data_processor.process_data(train_data, test_data)
    
    # 创建embedding管理器
    print("\n初始化Embedding管理器...")
    embedding_manager = EmbeddingManager(config, data_processor.word_to_idx)
    
    # 创建embedding层
    embedding_layer = embedding_manager.create_embedding_layer(trainable=True)
    
    # 确保配置中的embedding_dim与实际embedding维度一致
    if config.embedding == 'glove':
        config.embedding_dim = config.glove_dim
    
    # 创建模型
    model = create_model(config, embedding_layer)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数总数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # 创建训练器
    trainer = Trainer(config, model, train_loader, val_loader, test_loader)
    
    # 训练模型
    trainer.train()
    
    # 测试模型
    test_accuracy, predictions, labels = trainer.test()
    
    # 保存模型和结果
    if config.save_model:
        # 创建保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{config.model}_{config.embedding}_{timestamp}"
        
        # 保存模型
        model_path = os.path.join(config.model_dir, f"{model_name}.pth")
        trainer.save_model(model_path)
        
        # 保存词汇表
        vocab_path = os.path.join(config.model_dir, f"{model_name}_vocab.pkl")
        data_processor.save_vocabulary(vocab_path)
        
        # 保存训练结果
        results_path = os.path.join(config.model_dir, f"{model_name}_results.json")
        trainer.save_results(results_path)
        
        # 保存训练曲线
        curves_path = os.path.join(config.model_dir, f"{model_name}_curves.png")
        trainer.plot_training_curves(curves_path)
        
        # 自动预测test.tsv文件
        test_file_path = "../lab1/data/test.tsv"
        prediction_output = os.path.join(config.model_dir, f"{model_name}_predictions.csv")
        
        # 预测test.tsv
        predict_test_file(model, data_processor, config, test_file_path, prediction_output)
    
    print(f"\n最终测试准确率: {test_accuracy:.2f}%")
    print("训练完成！")

if __name__ == "__main__":
    main()