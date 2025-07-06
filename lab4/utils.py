"""
工具函数：数据处理、评估指标等
"""

import os
import json
import torch
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_conll_data(file_path):
    """
    加载CONLL格式的数据
    """
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split()
                if len(parts) >= 4:  # CONLL格式：词 词性 块标签 实体标签
                    word = parts[0]
                    pos = parts[1]
                    chunk = parts[2]
                    ner = parts[3]
                    current_sentence.append((word, pos, chunk, ner))
    
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences


def build_vocab(sentences, config):
    """
    构建词汇表
    """
    word_freq = Counter()
    
    for sentence in sentences:
        for word, _, _, _ in sentence:
            word_freq[word.lower()] += 1
    
    # 构建词汇表
    vocab = {config.pad_token: 0, config.unk_token: 1, config.start_token: 2, config.end_token: 3}
    
    # 按频率排序，只保留频率>=min_freq的词
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    for word, freq in sorted_words:
        if freq >= config.min_freq and len(vocab) < config.max_vocab_size:
            vocab[word] = len(vocab)
    
    return vocab


def build_label_vocab(sentences):
    """
    构建标签词汇表
    """
    label_set = set()
    for sentence in sentences:
        for _, _, _, ner in sentence:
            label_set.add(ner)
    
    label_vocab = {label: idx for idx, label in enumerate(sorted(label_set))}
    return label_vocab


def sentence_to_ids(sentence, vocab, config):
    """
    将句子转换为ID序列
    """
    word_ids = [vocab.get(config.start_token, vocab[config.unk_token])]
    
    for word, _, _, _ in sentence:
        word_id = vocab.get(word.lower(), vocab[config.unk_token])
        word_ids.append(word_id)
    
    word_ids.append(vocab.get(config.end_token, vocab[config.unk_token]))
    return word_ids


def sentence_to_labels(sentence, label_vocab):
    """
    将句子标签转换为ID序列
    """
    label_ids = [label_vocab['O']]  # START标签
    
    for _, _, _, ner in sentence:
        label_ids.append(label_vocab[ner])
    
    label_ids.append(label_vocab['O'])  # END标签
    return label_ids


def pad_sequences(sequences, pad_value=0):
    """
    填充序列到相同长度
    """
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    masks = []
    
    for seq in sequences:
        padded_seq = seq + [pad_value] * (max_len - len(seq))
        mask = [1] * len(seq) + [0] * (max_len - len(seq))
        padded_sequences.append(padded_seq)
        masks.append(mask)
    
    return padded_sequences, masks


def compute_metrics(y_true, y_pred, label_names):
    """
    计算评估指标
    """
    # 移除特殊标签（START, END, PAD）
    valid_indices = [i for i, label in enumerate(y_true) if label != 0]  # 假设0是PAD标签
    
    if not valid_indices:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]
    
    precision = precision_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
    recall = recall_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
    f1 = f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_training_curves(train_losses, dev_losses, train_f1s, dev_f1s, save_path):
    """
    绘制训练曲线
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(dev_losses, label='Dev Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # F1分数曲线
    ax2.plot(train_f1s, label='Train F1', color='blue')
    ax2.plot(dev_f1s, label='Dev F1', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Training and Validation F1 Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results(results, save_path):
    """
    保存结果到JSON文件
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results(load_path):
    """
    从JSON文件加载结果
    """
    with open(load_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_directories(config):
    """
    创建必要的目录
    """
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)


def set_random_seed(seed):
    """
    设置随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 