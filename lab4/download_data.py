"""
数据下载脚本：下载CONLL 2003 NER数据集
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path


def download_file(url, filename):
    """
    下载文件
    """
    print(f"Downloading {filename} from {url}...")
    
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Download completed: {filename}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """
    解压ZIP文件
    """
    print(f"Extracting {zip_path} to {extract_to}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extraction completed")
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def setup_data_directory():
    """
    设置数据目录
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir


def download_conll2003():
    """
    下载CONLL 2003数据集
    """
    print("=" * 60)
    print("Downloading CONLL 2003 NER Dataset")
    print("=" * 60)
    
    # 设置数据目录
    data_dir = setup_data_directory()
    
    # CONLL 2003数据集URL
    conll_url = "https://www.clips.uantwerpen.be/conll2003/ner.tgz"
    zip_filename = "ner.tgz"
    zip_path = data_dir / zip_filename
    
    # 检查是否已经存在
    if zip_path.exists():
        print(f"Dataset already exists at {zip_path}")
        response = input("Do you want to re-download? (y/n): ").lower()
        if response != 'y':
            print("Using existing dataset.")
            return True
    
    # 下载数据集
    if not download_file(conll_url, zip_path):
        print("Failed to download dataset.")
        return False
    
    # 解压数据集
    if not extract_zip(zip_path, data_dir):
        print("Failed to extract dataset.")
        return False
    
    # 移动文件到正确位置
    extracted_dir = data_dir / "ner"
    if extracted_dir.exists():
        # 移动训练文件
        train_file = extracted_dir / "eng.train"
        if train_file.exists():
            shutil.move(str(train_file), str(data_dir / "eng.train"))
        
        # 移动验证文件
        dev_file = extracted_dir / "eng.testa"
        if dev_file.exists():
            shutil.move(str(dev_file), str(data_dir / "eng.testa"))
        
        # 移动测试文件
        test_file = extracted_dir / "eng.testb"
        if test_file.exists():
            shutil.move(str(test_file), str(data_dir / "eng.testb"))
        
        # 删除临时目录
        shutil.rmtree(extracted_dir)
    
    # 删除ZIP文件
    if zip_path.exists():
        zip_path.unlink()
    
    print("Dataset setup completed!")
    return True


def verify_data_files():
    """
    验证数据文件
    """
    data_dir = Path("data")
    required_files = ["eng.train", "eng.testa", "eng.testb"]
    
    print("\nVerifying data files...")
    all_exist = True
    
    for filename in required_files:
        file_path = data_dir / filename
        if file_path.exists():
            file_size = file_path.stat().st_size
            print(f"✓ {filename} ({file_size:,} bytes)")
        else:
            print(f"✗ {filename} (missing)")
            all_exist = False
    
    if all_exist:
        print("\nAll data files are present and ready for use!")
        return True
    else:
        print("\nSome data files are missing. Please check the download.")
        return False


def show_data_info():
    """
    显示数据信息
    """
    data_dir = Path("data")
    
    print("\nData Information:")
    print("-" * 40)
    
    for filename in ["eng.train", "eng.testa", "eng.testb"]:
        file_path = data_dir / filename
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 统计句子数量
            sentences = 0
            for line in lines:
                if line.strip() == '':
                    sentences += 1
            
            print(f"{filename}:")
            print(f"  Lines: {len(lines):,}")
            print(f"  Sentences: {sentences:,}")
            print(f"  Size: {file_path.stat().st_size:,} bytes")
            print()


def main():
    """
    主函数
    """
    print("CONLL 2003 NER Dataset Downloader")
    print("=" * 60)
    print("1. Download dataset")
    print("2. Verify data files")
    print("3. Show data information")
    print("4. Exit")
    
    while True:
        choice = input("\nPlease select an option (1-4): ").strip()
        
        if choice == '1':
            success = download_conll2003()
            if success:
                verify_data_files()
            break
        elif choice == '2':
            verify_data_files()
            break
        elif choice == '3':
            show_data_info()
            break
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-4.")


if __name__ == "__main__":
    main() 