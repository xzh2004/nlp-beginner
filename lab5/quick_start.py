#!/usr/bin/env python3
"""
快速启动脚本
"""

import torch
import sys
import os

def main():
    print("欢迎使用基于神经网络的语言模型！")
    print("本项目实现了LSTM和GRU字符级语言模型。\n")
    
    print("=== 使用示例 ===")
    print("1. 快速训练LSTM模型:")
    print("   python train.py --model_type lstm --num_epochs 5 --batch_size 16")
    print("\n2. 快速训练GRU模型:")
    print("   python train.py --model_type gru --num_epochs 5 --batch_size 16")
    print("\n3. 生成唐诗:")
    print("   python generate.py --model_path models/lstm_YYYYMMDD_HHMMSS/best_model.pth --num_poems 3")
    
    print("\n=== 开始使用 ===")
    print("建议先运行: python test_basic.py 检查环境")

if __name__ == '__main__':
    main() 