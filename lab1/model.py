import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime

class LossFunctions:
    """
    损失函数集合类
    支持多种损失函数：sigmoid, relu, hinge, zero-one, cross_entropy
    """
    
    @staticmethod
    def sigmoid_loss(y_pred, y_true, epsilon=1e-15):
        """
        Sigmoid损失函数
        对于二分类，使用sigmoid激活函数后的交叉熵损失
        L = -log(sigmoid(y_pred)) if y_true=1, -log(1-sigmoid(y_pred)) if y_true=0
        """
        # 首先应用sigmoid激活函数
        y_pred_sigmoid = 1 / (1 + np.exp(-y_pred))
        # 防止数值溢出
        y_pred_sigmoid = np.clip(y_pred_sigmoid, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred_sigmoid) + (1 - y_true) * np.log(1 - y_pred_sigmoid))
    
    @staticmethod
    def sigmoid_gradient(y_pred, y_true):
        """
        Sigmoid损失函数的梯度
        """
        # 首先应用sigmoid激活函数
        y_pred_sigmoid = 1 / (1 + np.exp(-y_pred))
        return y_pred_sigmoid - y_true
    
    @staticmethod
    def relu_loss(y_pred, y_true):
        """
        真正的ReLU损失函数
        L = max(0, -y_true * y_pred)
        当y_true=1时，L = max(0, -y_pred)
        当y_true=0时，L = max(0, 0) = 0
        """
        return np.mean(np.maximum(0, -y_true * y_pred))
    
    @staticmethod
    def relu_gradient(y_pred, y_true):
        """
        ReLU损失函数的梯度
        """
        gradient = np.where(-y_true * y_pred > 0, -y_true, 0)
        return gradient
    
    @staticmethod
    def hinge_loss(y_pred, y_true):
        """
        Hinge损失函数
        L = max(0, 1 - y_true * y_pred)
        """
        # 将y_true从{0,1}转换为{-1,1}
        y_true_transformed = 2 * y_true - 1
        margin = 1 - y_true_transformed * y_pred
        return np.mean(np.maximum(0, margin))
    
    @staticmethod
    def hinge_gradient(y_pred, y_true):
        """
        Hinge损失函数的梯度
        """
        # 将y_true从{0,1}转换为{-1,1}
        y_true_transformed = 2 * y_true - 1
        margin = 1 - y_true_transformed * y_pred
        gradient = np.where(margin > 0, -y_true_transformed, 0)
        return gradient
    
    @staticmethod
    def zero_one_loss(y_pred, y_true, threshold=0.5):
        """
        Zero-One损失函数
        L = 1 if y_pred != y_true, 0 otherwise
        """
        y_pred_binary = (y_pred >= threshold).astype(int)
        return np.mean(y_pred_binary != y_true)
    
    @staticmethod
    def zero_one_gradient(y_pred, y_true, threshold=0.5):
        """
        Zero-One损失函数的梯度（近似）
        由于zero-one损失不可微，这里使用一个可微的近似
        """
        # 使用sigmoid的梯度作为近似
        return y_pred - y_true
    
    @staticmethod
    def cross_entropy_loss(y_pred, y_true, epsilon=1e-15):
        """
        交叉熵损失函数
        L = -sum(y_true * log(y_pred))
        """
        # 避免log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def cross_entropy_gradient(y_pred, y_true):
        """
        交叉熵损失函数的梯度
        """
        return y_pred - y_true
    
    @staticmethod
    def get_loss_function(loss_type):
        """
        根据损失函数类型返回对应的损失函数和梯度函数
        """
        loss_functions = {
            'sigmoid': (LossFunctions.sigmoid_loss, LossFunctions.sigmoid_gradient),
            'relu': (LossFunctions.relu_loss, LossFunctions.relu_gradient),
            'hinge': (LossFunctions.hinge_loss, LossFunctions.hinge_gradient),
            'zero_one': (LossFunctions.zero_one_loss, LossFunctions.zero_one_gradient),
            'cross_entropy': (LossFunctions.cross_entropy_loss, LossFunctions.cross_entropy_gradient)
        }
        
        if loss_type not in loss_functions:
            raise ValueError(f"Unsupported loss function: {loss_type}. "
                           f"Supported types: {list(loss_functions.keys())}")
        
        return loss_functions[loss_type]
    
    @staticmethod
    def multiclass_cross_entropy_loss(y_pred, y_true, epsilon=1e-15):
        """
        多分类交叉熵损失函数
        L = -sum(y_true * log(y_pred))
        """
        # 避免log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def multiclass_cross_entropy_gradient(y_pred, y_true):
        """
        多分类交叉熵损失函数的梯度
        """
        return y_pred - y_true
    
    @staticmethod
    def multiclass_hinge_loss(y_pred, y_true):
        """
        多分类Hinge损失函数
        L = max(0, 1 + max(y_pred_wrong) - y_pred_correct)
        """
        # 计算正确类别的预测值
        correct_pred = np.sum(y_pred * y_true, axis=1, keepdims=True)
        
        # 计算错误类别的最大预测值
        wrong_pred = y_pred * (1 - y_true)
        wrong_pred = np.where(wrong_pred == 0, -np.inf, wrong_pred)
        max_wrong_pred = np.max(wrong_pred, axis=1, keepdims=True)
        
        # 计算hinge损失
        margin = 1 + max_wrong_pred - correct_pred
        return np.mean(np.maximum(0, margin))
    
    @staticmethod
    def multiclass_hinge_gradient(y_pred, y_true):
        """
        多分类Hinge损失函数的梯度（近似）
        """
        # 使用交叉熵梯度作为近似
        return y_pred - y_true
    
    @staticmethod
    def multiclass_relu_loss(y_pred, y_true):
        """
        多分类ReLU损失函数
        L = max(0, max(y_pred_wrong) - y_pred_correct)
        当正确类别的预测值大于所有错误类别的预测值时，损失为0
        否则损失为最大错误预测值与正确预测值的差
        """
        # 计算正确类别的预测值
        correct_pred = np.sum(y_pred * y_true, axis=1, keepdims=True)
        
        # 计算错误类别的最大预测值
        wrong_pred = y_pred * (1 - y_true)
        wrong_pred = np.where(wrong_pred == 0, -np.inf, wrong_pred)
        max_wrong_pred = np.max(wrong_pred, axis=1, keepdims=True)
        
        # 计算ReLU损失
        margin = max_wrong_pred - correct_pred
        return np.mean(np.maximum(0, margin))
    
    @staticmethod
    def multiclass_relu_gradient(y_pred, y_true):
        """
        多分类ReLU损失函数的梯度
        当margin > 0时，梯度为错误类别最大预测值对应的梯度
        当margin <= 0时，梯度为0
        """
        # 计算正确类别的预测值
        correct_pred = np.sum(y_pred * y_true, axis=1, keepdims=True)
        
        # 计算错误类别的最大预测值
        wrong_pred = y_pred * (1 - y_true)
        wrong_pred = np.where(wrong_pred == 0, -np.inf, wrong_pred)
        max_wrong_pred = np.max(wrong_pred, axis=1, keepdims=True)
        
        # 计算margin
        margin = max_wrong_pred - correct_pred
        
        # 计算梯度
        gradient = np.zeros_like(y_pred)
        
        # 当margin > 0时，需要更新梯度
        active_samples = margin > 0
        
        if np.any(active_samples):
            # 对于激活的样本，找到最大错误预测值对应的类别
            for i in np.where(active_samples)[0]:
                wrong_mask = (1 - y_true[i]) > 0
                if np.any(wrong_mask):
                    wrong_pred_i = y_pred[i][wrong_mask]
                    max_wrong_idx = np.argmax(wrong_pred_i)
                    # 找到原始索引
                    wrong_indices = np.where(wrong_mask)[0]
                    max_wrong_original_idx = wrong_indices[max_wrong_idx]
                    
                    # 更新梯度：增加错误类别的预测值，减少正确类别的预测值
                    gradient[i][max_wrong_original_idx] = 1
                    gradient[i] -= y_true[i]
        
        return gradient
    
    @staticmethod
    def get_multiclass_loss_function(loss_type):
        """
        根据损失函数类型返回对应的多分类损失函数和梯度函数
        """
        multiclass_loss_functions = {
            'cross_entropy': (LossFunctions.multiclass_cross_entropy_loss, 
                             LossFunctions.multiclass_cross_entropy_gradient),
            'sigmoid': (LossFunctions.multiclass_sigmoid_loss, 
                       LossFunctions.multiclass_sigmoid_gradient),
            'hinge': (LossFunctions.multiclass_hinge_loss, 
                     LossFunctions.multiclass_hinge_gradient),
            'relu': (LossFunctions.multiclass_relu_loss, 
                    LossFunctions.multiclass_relu_gradient)
        }
        
        if loss_type not in multiclass_loss_functions:
            # 对于不支持的损失函数，默认使用交叉熵
            print(f"Warning: {loss_type} not supported for multiclass, using cross_entropy instead")
            return multiclass_loss_functions['cross_entropy']
        
        return multiclass_loss_functions[loss_type]
    
    @staticmethod
    def multiclass_sigmoid_loss(y_pred, y_true, epsilon=1e-15):
        """
        多分类Sigmoid损失函数
        对每个类别应用sigmoid激活函数，然后计算交叉熵损失
        """
        # 对每个类别应用sigmoid激活函数
        y_pred_sigmoid = 1 / (1 + np.exp(-y_pred))
        # 防止数值溢出
        y_pred_sigmoid = np.clip(y_pred_sigmoid, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred_sigmoid) + (1 - y_true) * np.log(1 - y_pred_sigmoid), axis=1))
    
    @staticmethod
    def multiclass_sigmoid_gradient(y_pred, y_true):
        """
        多分类Sigmoid损失函数的梯度
        """
        # 对每个类别应用sigmoid激活函数
        y_pred_sigmoid = 1 / (1 + np.exp(-y_pred))
        return y_pred_sigmoid - y_true

class LogisticRegression:
    """
    逻辑回归（二分类）
    """
    def __init__(self, learning_rate=0.01, max_iter=1000, batch_size=32, 
                 regularization='l2', lambda_reg=0.01, early_stopping=True, 
                 patience=10, random_state=42, plot_training=True, config=None,
                 shuffle=True, batch_type='mini_batch', loss_function='cross_entropy'):
        """
        初始化逻辑回归模型
        Args:
            learning_rate: 学习率
            max_iter: 最大迭代次数
            batch_size: 批量大小
            regularization: 正则化类型 ('l1', 'l2', None)
            lambda_reg: 正则化强度
            early_stopping: 是否使用早停
            patience: 早停耐心值
            random_state: 随机种子
            plot_training: 是否绘制训练曲线
            config: 模型配置字典
            shuffle: 是否打乱数据
            batch_type: 批量类型 ('mini_batch', 'batch', 'stochastic')
            loss_function: 损失函数类型 ('sigmoid', 'relu', 'hinge', 'zero_one', 'cross_entropy')
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state
        self.plot_training = plot_training
        self.config = config
        self.shuffle = shuffle
        self.batch_type = batch_type
        self.loss_function = loss_function
        
        # 从配置中获取损失函数（如果配置存在）
        if config and 'model' in config and 'loss_function' in config['model']:
            self.loss_function = config['model']['loss_function']
        
        # 获取损失函数和梯度函数
        self.loss_func, self.gradient_func = LossFunctions.get_loss_function(self.loss_function)
        
        self.weights = None
        self.bias = None
        self.training_history = []
        
        np.random.seed(random_state)
    
    def sigmoid(self, z):
        """
        Sigmoid激活函数
        """
        # 防止数值溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def initialize_parameters(self, n_features):
        """
        初始化参数
        """
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
    
    def forward(self, X):
        """
        前向传播
        """
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def compute_loss(self, y_pred, y_true):
        """
        计算损失函数
        """
        # 使用配置的损失函数
        loss = self.loss_func(y_pred, y_true)
        
        # 正则化损失
        if self.regularization == 'l2':
            reg_loss = self.lambda_reg * np.sum(self.weights ** 2) / 2
            loss += reg_loss
        elif self.regularization == 'l1':
            reg_loss = self.lambda_reg * np.sum(np.abs(self.weights))
            loss += reg_loss
        
        return loss
    
    def compute_gradients(self, X, y_pred, y_true):
        """
        计算梯度
        """
        m = X.shape[0]
        
        # 使用配置的梯度函数
        dz = self.gradient_func(y_pred, y_true)
        dw = np.dot(X.T, dz) / m
        db = np.mean(dz)
        
        # 正则化梯度
        if self.regularization == 'l2':
            dw += self.lambda_reg * self.weights
        elif self.regularization == 'l1':
            dw += self.lambda_reg * np.sign(self.weights)
        
        return dw, db
    
    def create_batches(self, X, y):
        """
        创建批量数据
        支持三种模式：
        - mini_batch: 小批量梯度下降
        - batch: 全批量梯度下降
        - stochastic: 随机梯度下降
        """
        m = X.shape[0]
        
        # 根据batch_type调整batch_size
        if self.batch_type == 'batch':
            # 全批量：使用所有数据
            batch_size = m
        elif self.batch_type == 'stochastic':
            # 随机梯度下降：每次只用一个样本
            batch_size = 1
        else:
            # mini_batch：使用指定的batch_size
            batch_size = self.batch_size
        
        # 生成索引
        if self.shuffle:
            indices = np.random.permutation(m)
        else:
            indices = np.arange(m)
        
        # 创建批量
        for i in range(0, m, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield X[batch_indices], y[batch_indices]
    
    def plot_training_curves(self):
        """
        绘制训练曲线
        """
        if not self.training_history:
            print("No training history available for plotting.")
            return
        
        epochs = [h['epoch'] for h in self.training_history]
        train_losses = [h['train_loss'] for h in self.training_history]
        
        # 创建图形，增加高度以容纳配置信息
        fig = plt.figure(figsize=(15, 8))
        
        # 添加配置信息文本
        if self.config:
            config_text = self._format_config_info()
            fig.suptitle('Training Curves with Model Configuration', fontsize=16, fontweight='bold')
            
            # 在图形顶部添加配置信息
            plt.figtext(0.02, 0.95, config_text, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                       verticalalignment='top', fontfamily='monospace')
        else:
            fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
        
        # 训练损失
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 验证损失（如果有）
        if 'val_loss' in self.training_history[0]:
            val_losses = [h['val_loss'] for h in self.training_history]
            plt.subplot(1, 3, 2)
            plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
            plt.title('Training vs Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 验证准确率（如果有）
        if 'val_acc' in self.training_history[0]:
            val_accs = [h['val_acc'] for h in self.training_history]
            plt.subplot(1, 3, 3)
            plt.plot(epochs, val_accs, 'g-', label='Validation Accuracy', linewidth=2)
            plt.title('Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # 为配置信息留出空间
        
        # 创建图片保存目录
        plots_dir = 'training_plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # 生成描述性文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config and 'model' in self.config:
            model_config = self.config['model']
            lr = model_config.get('learning_rate', 'unknown_lr')
            reg = model_config.get('regularization', 'none')
            lambda_reg = model_config.get('lambda_reg', 'unknown_lambda')
            
            # 清理文件名中的特殊字符
            lr_str = str(lr).replace('.', 'p').replace('-', 'neg')
            lambda_str = str(lambda_reg).replace('.', 'p').replace('-', 'neg')
            
            filename = f'training_curves_logistic_lr{lr_str}_reg{reg}_lambda{lambda_str}_{timestamp}.png'
        else:
            filename = f'training_curves_logistic_{timestamp}.png'
        
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training curves saved as '{filepath}'")
    
    def _format_config_info(self):
        """
        格式化配置信息用于显示
        """
        if not self.config:
            return "No configuration available"
        
        config_lines = []
        config_lines.append("Model Configuration:")
        config_lines.append("=" * 50)
        
        # 特征配置
        if 'features' in self.config:
            features = self.config['features']
            config_lines.append("Features:")
            config_lines.append(f"  Type: {features.get('type', 'N/A')}")
            config_lines.append(f"  Max Features: {features.get('max_features', 'N/A')}")
            config_lines.append(f"  Min DF: {features.get('min_df', 'N/A')}")
            config_lines.append(f"  Max DF: {features.get('max_df', 'N/A')}")
            config_lines.append(f"  N-gram Range: {features.get('ngram_range', 'N/A')}")
            config_lines.append(f"  Normalize: {features.get('normalize', 'N/A')}")
            
            if features.get('feature_selection', {}).get('enabled', False):
                fs_config = features['feature_selection']
                config_lines.append(f"  Feature Selection: {fs_config.get('method', 'N/A')} (k={fs_config.get('k', 'N/A')})")
            else:
                config_lines.append("  Feature Selection: Disabled")
        
        config_lines.append("")
        
        # 模型配置
        if 'model' in self.config:
            model = self.config['model']
            config_lines.append("Model:")
            config_lines.append(f"  Type: {model.get('type', 'N/A')}")
            config_lines.append(f"  Loss Function: {model.get('loss_function', 'cross_entropy')}")
            config_lines.append(f"  Learning Rate: {model.get('learning_rate', 'N/A')}")
            config_lines.append(f"  Max Iterations: {model.get('max_iter', 'N/A')}")
            config_lines.append(f"  Batch Size: {model.get('batch_size', 'N/A')}")
            config_lines.append(f"  Batch Type: {model.get('batch_type', 'mini_batch')}")
            config_lines.append(f"  Shuffle: {model.get('shuffle', True)}")
            
            reg_type = model.get('regularization', 'None')
            reg_lambda = model.get('lambda_reg', 'N/A')
            if reg_type:
                config_lines.append(f"  Regularization: {reg_type} (λ={reg_lambda})")
            else:
                config_lines.append("  Regularization: None")
            
            config_lines.append(f"  Early Stopping: {model.get('early_stopping', 'N/A')}")
            if model.get('early_stopping', False):
                config_lines.append(f"  Patience: {model.get('patience', 'N/A')}")
        
        return "\n".join(config_lines)
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        训练模型
        """
        print("Training Logistic Regression...")
        
        # 初始化参数
        if self.weights is None:
            self.initialize_parameters(X.shape[1])
        
        # 早停相关变量
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_iter):
            epoch_loss = 0
            num_batches = 0
            
            # 创建小批量
            for X_batch, y_batch in self.create_batches(X, y):
                # 前向传播
                y_pred = self.forward(X_batch)
                
                # 计算损失
                batch_loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += batch_loss
                num_batches += 1
                
                # 计算梯度
                dw, db = self.compute_gradients(X_batch, y_pred, y_batch)
                
                # 更新参数
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            # 计算平均损失
            avg_loss = epoch_loss / num_batches
            
            # 记录训练历史
            history = {'epoch': epoch, 'train_loss': avg_loss}
            
            # 验证集评估
            if X_val is not None and y_val is not None:
                val_pred = self.predict_proba(X_val)
                val_loss = self.compute_loss(val_pred, y_val)
                val_acc = accuracy_score(y_val, self.predict(X_val))
                history['val_loss'] = val_loss
                history['val_acc'] = val_acc
                
                # 早停检查
                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            self.training_history.append(history)
            
            # 打印进度
            if epoch % 100 == 0 or epoch == self.max_iter - 1:
                if X_val is not None and y_val is not None:
                    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
        
        print("Training completed!")
        
        # 绘制训练曲线
        if self.plot_training:
            self.plot_training_curves()
        
        return self
    
    def predict_proba(self, X):
        """
        预测概率
        """
        return self.forward(X)
    
    def predict(self, X, threshold=0.5):
        """
        预测类别
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def evaluate(self, X, y):
        """
        评估模型
        """
        # 预测X的标签
        y_pred = self.predict(X)
        # 计算准确率
        accuracy = accuracy_score(y, y_pred)
        # 计算精确率、召回率和F1分数
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
        
        # 返回评估结果
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

class SoftmaxRegression:
    """
    Softmax回归（多分类）
    """
    def __init__(self, n_classes, learning_rate=0.01, max_iter=1000, batch_size=32,
                 regularization='l2', lambda_reg=0.01, early_stopping=True,
                 patience=10, random_state=42, plot_training=True, config=None,
                 shuffle=True, batch_type='mini_batch', loss_function='cross_entropy'):
        """
        初始化Softmax回归模型
        Args:
            n_classes: 类别数量
            learning_rate: 学习率
            max_iter: 最大迭代次数
            batch_size: 批量大小
            regularization: 正则化类型 ('l1', 'l2', None)
            lambda_reg: 正则化强度
            early_stopping: 是否使用早停
            patience: 早停耐心值
            random_state: 随机种子
            plot_training: 是否绘制训练曲线
            config: 模型配置字典
            shuffle: 是否打乱数据
            batch_type: 批量类型 ('mini_batch', 'batch', 'stochastic')
            loss_function: 损失函数类型 ('sigmoid', 'relu', 'hinge', 'zero_one', 'cross_entropy')
        """
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state
        self.plot_training = plot_training
        self.config = config
        self.shuffle = shuffle
        self.batch_type = batch_type
        self.loss_function = loss_function
        
        # 从配置中获取损失函数（如果配置存在）
        if config and 'model' in config and 'loss_function' in config['model']:
            self.loss_function = config['model']['loss_function']
        
        # 获取损失函数和梯度函数
        self.loss_func, self.gradient_func = LossFunctions.get_multiclass_loss_function(self.loss_function)
        
        self.weights = None
        self.training_history = []
        
        np.random.seed(random_state)
    
    def softmax(self, z):
        """
        Softmax函数
        """
        # 防止数值溢出
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def initialize_parameters(self, n_features):
        """
        初始化参数
        """
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
    
    def forward(self, X):
        """
        前向传播
        """
        z = np.dot(X, self.weights)
        return self.softmax(z)
    
    def compute_loss(self, y_pred, y_true):
        """
        计算损失函数
        """
        # 使用配置的损失函数
        loss = self.loss_func(y_pred, y_true)
        
        # 正则化损失
        if self.regularization == 'l2':
            reg_loss = self.lambda_reg * np.sum(self.weights ** 2) / 2
            loss += reg_loss
        elif self.regularization == 'l1':
            reg_loss = self.lambda_reg * np.sum(np.abs(self.weights))
            loss += reg_loss
        
        return loss
    
    def compute_gradients(self, X, y_pred, y_true):
        """
        计算梯度
        """
        m = X.shape[0]
        
        # 使用配置的梯度函数
        dz = self.gradient_func(y_pred, y_true)
        dw = np.dot(X.T, dz) / m
        
        # 正则化梯度
        if self.regularization == 'l2':
            dw += self.lambda_reg * self.weights
        elif self.regularization == 'l1':
            dw += self.lambda_reg * np.sign(self.weights)
        
        return dw
    
    def create_batches(self, X, y):
        """
        创建批量数据
        支持三种模式：
        - mini_batch: 小批量梯度下降
        - batch: 全批量梯度下降
        - stochastic: 随机梯度下降
        """
        m = X.shape[0]
        
        # 根据batch_type调整batch_size
        if self.batch_type == 'batch':
            # 全批量：使用所有数据
            batch_size = m
        elif self.batch_type == 'stochastic':
            # 随机梯度下降：每次只用一个样本
            batch_size = 1
        else:
            # mini_batch：使用指定的batch_size
            batch_size = self.batch_size
        
        # 生成索引
        if self.shuffle:
            indices = np.random.permutation(m)
        else:
            indices = np.arange(m)
        
        # 创建批量
        for i in range(0, m, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield X[batch_indices], y[batch_indices]
    
    def plot_training_curves(self):
        """
        绘制训练曲线
        """
        if not self.training_history:
            print("No training history available for plotting.")
            return
        
        epochs = [h['epoch'] for h in self.training_history]
        train_losses = [h['train_loss'] for h in self.training_history]
        
        # 创建图形，增加高度以容纳配置信息
        fig = plt.figure(figsize=(15, 8))
        
        # 添加配置信息文本
        if self.config:
            config_text = self._format_config_info()
            fig.suptitle('Training Curves with Model Configuration', fontsize=16, fontweight='bold')
            
            # 在图形顶部添加配置信息
            plt.figtext(0.02, 0.95, config_text, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                       verticalalignment='top', fontfamily='monospace')
        else:
            fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
        
        # 训练损失
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 验证损失（如果有）
        if 'val_loss' in self.training_history[0]:
            val_losses = [h['val_loss'] for h in self.training_history]
            plt.subplot(1, 3, 2)
            plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
            plt.title('Training vs Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 验证准确率（如果有）
        if 'val_acc' in self.training_history[0]:
            val_accs = [h['val_acc'] for h in self.training_history]
            plt.subplot(1, 3, 3)
            plt.plot(epochs, val_accs, 'g-', label='Validation Accuracy', linewidth=2)
            plt.title('Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # 为配置信息留出空间
        
        # 创建图片保存目录
        plots_dir = 'training_plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # 生成描述性文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config and 'model' in self.config:
            model_config = self.config['model']
            lr = model_config.get('learning_rate', 'unknown_lr')
            reg = model_config.get('regularization', 'none')
            lambda_reg = model_config.get('lambda_reg', 'unknown_lambda')
            
            # 清理文件名中的特殊字符
            lr_str = str(lr).replace('.', 'p').replace('-', 'neg')
            lambda_str = str(lambda_reg).replace('.', 'p').replace('-', 'neg')
            
            filename = f'training_curves_softmax_lr{lr_str}_reg{reg}_lambda{lambda_str}_{timestamp}.png'
        else:
            filename = f'training_curves_softmax_{timestamp}.png'
        
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training curves saved as '{filepath}'")
    
    def _format_config_info(self):
        """
        格式化配置信息用于显示
        """
        if not self.config:
            return "No configuration available"
        
        config_lines = []
        config_lines.append("Model Configuration:")
        config_lines.append("=" * 50)
        
        # 特征配置
        if 'features' in self.config:
            features = self.config['features']
            config_lines.append("Features:")
            config_lines.append(f"  Type: {features.get('type', 'N/A')}")
            config_lines.append(f"  Max Features: {features.get('max_features', 'N/A')}")
            config_lines.append(f"  Min DF: {features.get('min_df', 'N/A')}")
            config_lines.append(f"  Max DF: {features.get('max_df', 'N/A')}")
            config_lines.append(f"  N-gram Range: {features.get('ngram_range', 'N/A')}")
            config_lines.append(f"  Normalize: {features.get('normalize', 'N/A')}")
            
            if features.get('feature_selection', {}).get('enabled', False):
                fs_config = features['feature_selection']
                config_lines.append(f"  Feature Selection: {fs_config.get('method', 'N/A')} (k={fs_config.get('k', 'N/A')})")
            else:
                config_lines.append("  Feature Selection: Disabled")
        
        config_lines.append("")
        
        # 模型配置
        if 'model' in self.config:
            model = self.config['model']
            config_lines.append("Model:")
            config_lines.append(f"  Type: {model.get('type', 'N/A')}")
            config_lines.append(f"  Loss Function: {model.get('loss_function', 'cross_entropy')}")
            config_lines.append(f"  Learning Rate: {model.get('learning_rate', 'N/A')}")
            config_lines.append(f"  Max Iterations: {model.get('max_iter', 'N/A')}")
            config_lines.append(f"  Batch Size: {model.get('batch_size', 'N/A')}")
            config_lines.append(f"  Batch Type: {model.get('batch_type', 'mini_batch')}")
            config_lines.append(f"  Shuffle: {model.get('shuffle', True)}")
            
            reg_type = model.get('regularization', 'None')
            reg_lambda = model.get('lambda_reg', 'N/A')
            if reg_type:
                config_lines.append(f"  Regularization: {reg_type} (λ={reg_lambda})")
            else:
                config_lines.append("  Regularization: None")
            
            config_lines.append(f"  Early Stopping: {model.get('early_stopping', 'N/A')}")
            if model.get('early_stopping', False):
                config_lines.append(f"  Patience: {model.get('patience', 'N/A')}")
        
        return "\n".join(config_lines)
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        训练模型
        """
        print("Training Softmax Regression...")
        
        # 将标签转换为one-hot编码
        y_onehot = self._to_onehot(y)
        if y_val is not None:
            y_val_onehot = self._to_onehot(y_val)
        
        # 初始化参数
        if self.weights is None:
            self.initialize_parameters(X.shape[1])
        
        # 早停相关变量
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_iter):
            epoch_loss = 0
            num_batches = 0
            
            # 创建小批量
            for X_batch, y_batch in self.create_batches(X, y_onehot):
                # 前向传播
                y_pred = self.forward(X_batch)
                
                # 计算损失
                batch_loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += batch_loss
                num_batches += 1
                
                # 计算梯度
                dw = self.compute_gradients(X_batch, y_pred, y_batch)
                
                # 更新参数
                self.weights -= self.learning_rate * dw
            
            # 计算平均损失
            avg_loss = epoch_loss / num_batches
            
            # 记录训练历史
            history = {'epoch': epoch, 'train_loss': avg_loss}
            
            # 验证集评估
            if X_val is not None and y_val is not None:
                val_pred = self.predict_proba(X_val)
                val_loss = self.compute_loss(val_pred, y_val_onehot)
                val_acc = accuracy_score(y_val, self.predict(X_val))
                history['val_loss'] = val_loss
                history['val_acc'] = val_acc
                
                # 早停检查
                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            self.training_history.append(history)
            
            # 打印进度
            if X_val is not None and y_val is not None:
                print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
            else:
                    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
        
        print("Training completed!")
        
        # 绘制训练曲线
        if self.plot_training:
            self.plot_training_curves()
        
        return self
    
    def predict_proba(self, X):
        """
        预测概率
        """
        return self.forward(X)
    
    def predict(self, X):
        """
        预测类别
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
    def evaluate(self, X, y):
        """
        评估模型
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _to_onehot(self, y):
        """
        将标签转换为one-hot编码
        """
        n_samples = len(y)
        y_onehot = np.zeros((n_samples, self.n_classes))
        y_onehot[np.arange(n_samples), y] = 1
        return y_onehot

def main():
    """
    测试模型
    """
    # 生成示例数据
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y_binary = (X[:, 0] + X[:, 1] > 0).astype(int)
    y_multi = np.random.randint(0, 3, 100)
    
    # 测试逻辑回归
    print("Testing Logistic Regression...")
    lr_model = LogisticRegression(learning_rate=0.1, max_iter=100, plot_training=True)
    lr_model.fit(X, y_binary)
    
    # 评估
    results = lr_model.evaluate(X, y_binary)
    print("Logistic Regression Results:", results)
    
    # 测试Softmax回归
    print("\nTesting Softmax Regression...")
    sm_model = SoftmaxRegression(n_classes=3, learning_rate=0.1, max_iter=100, plot_training=True)
    sm_model.fit(X, y_multi)
    
    # 评估
    results = sm_model.evaluate(X, y_multi)
    print("Softmax Regression Results:", results)

if __name__ == "__main__":
    main() 