#!/usr/bin/env python3
"""
Lab3 快速启动脚本
提供简化的训练和测试流程
"""

import os
import sys
import argparse
from config import Config
from train import main as train_main
from evaluate import main as evaluate_main
from demo import main as demo_main

def check_dependencies():
    """检查依赖是否安装"""
    try:
        import torch
        import numpy
        import pandas
        import tqdm
        import matplotlib
        import seaborn
        import sklearn
        print("✓ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def check_data():
    """检查数据文件是否存在"""
    config = Config()
    required_files = [
        config.train_path,
        config.dev_path,
        config.test_path
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("✗ 缺少数据文件:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("请确保SNLI数据集已下载到正确位置")
        return False
    
    print("✓ 数据文件检查通过")
    return True

def quick_train():
    """快速训练（使用较少数据）"""
    print("开始快速训练...")
    print("注意：这将使用较少的训练数据以加快速度")
    
    # 修改配置以使用较少数据
    config = Config()
    config.num_epochs = 5  # 减少训练轮数
    config.batch_size = 16  # 减少批次大小
    
    # 这里需要修改train.py中的参数
    print("请手动修改 train.py 中的以下参数:")
    print("  max_train_samples=5000  # 减少训练样本")
    print("  max_dev_samples=500     # 减少验证样本")
    print("然后运行: python train.py")

def quick_test():
    """快速测试"""
    print("开始快速测试...")
    
    if not os.path.exists("models/esim_model.pth"):
        print("✗ 模型文件不存在，请先训练模型")
        return False
    
    if not os.path.exists("models/vocab.pth"):
        print("✗ 词汇表文件不存在，请先训练模型")
        return False
    
    print("✓ 模型文件检查通过")
    return True

def main():
    parser = argparse.ArgumentParser(description="Lab3 快速启动脚本")
    parser.add_argument("--check", action="store_true", help="检查环境和数据")
    parser.add_argument("--train", action="store_true", help="开始训练")
    parser.add_argument("--test", action="store_true", help="开始测试")
    parser.add_argument("--demo", action="store_true", help="运行演示")
    parser.add_argument("--all", action="store_true", help="运行完整流程")
    
    args = parser.parse_args()
    
    print("="*50)
    print("Lab3: 基于注意力机制的文本匹配")
    print("="*50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查数据
    if not check_data():
        return
    
    if args.check:
        print("\n环境检查完成！")
        return
    
    if args.train or args.all:
        print("\n" + "="*30)
        print("开始训练流程")
        print("="*30)
        quick_train()
    
    if args.test or args.all:
        print("\n" + "="*30)
        print("开始测试流程")
        print("="*30)
        if quick_test():
            try:
                evaluate_main()
            except Exception as e:
                print(f"测试过程中出现错误: {e}")
    
    if args.demo or args.all:
        print("\n" + "="*30)
        print("开始演示流程")
        print("="*30)
        if quick_test():
            try:
                demo_main()
            except Exception as e:
                print(f"演示过程中出现错误: {e}")
    
    if not any([args.check, args.train, args.test, args.demo, args.all]):
        print("\n使用方法:")
        print("  python quick_start.py --check    # 检查环境")
        print("  python quick_start.py --train    # 开始训练")
        print("  python quick_start.py --test     # 开始测试")
        print("  python quick_start.py --demo     # 运行演示")
        print("  python quick_start.py --all      # 运行完整流程")

if __name__ == "__main__":
    main() 