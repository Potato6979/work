import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.outlier_indices = []
    
    def load_data(self, filepath):
        """加载原始数据"""
        df = pd.read_csv(filepath)
        print(f"数据加载完成: {df.shape[0]}行, {df.shape[1]}列")
        return df
    
    def basic_info(self, df):
        """基本数据信息"""
        print("=== 数据基本信息 ===")
        print(f"数据形状: {df.shape}")
        print(f"缺失值统计:\n{df.isnull().sum()}")
        print(f"数据类型:\n{df.dtypes}")
        print(f"目标变量分布:\n{df['target'].value_counts()}")
        return df.describe()
    
    def handle_missing_values(self, df):
        """处理缺失值"""
        print("=== 处理缺失值 ===")
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        for col in missing_cols:
            if df[col].dtype in ['int64', 'float64']:
                # 数值变量用中位数填充
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"{col}: 用中位数 {median_val} 填充 {df[col].isnull().sum()} 个缺失值")
            else:
                # 类别变量用众数填充
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"{col}: 用众数 {mode_val} 填充 {df[col].isnull().sum()} 个缺失值")
        
        print(f"处理后缺失值: {df.isnull().sum().sum()}")
        return df
    
    def detect_outliers(self, df, method='iqr'):
        """异常值检测和处理"""
        print("=== 异常值检测 ===")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        outlier_indices = set()
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                outlier_indices.update(col_outliers)
                print(f"{col}: 检测到 {len(col_outliers)} 个异常值")
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col]))
                col_outliers = df[z_scores > 3].index
                outlier_indices.update(col_outliers)
                print(f"{col}: 检测到 {len(col_outliers)} 个异常值")
        
        self.outlier_indices = list(outlier_indices)
        print(f"总异常值数量: {len(self.outlier_indices)}")
        
        # 删除异常值
        df_clean = df.drop(self.outlier_indices).reset_index(drop=True)
        print(f"清理后数据量: {df_clean.shape[0]}")
        return df_clean
    
    def feature_engineering(self, df):
        """特征工程"""
        print("=== 特征工程 ===")
        df_fe = df.copy()
        
        # 创建年龄分组
        df_fe['age_group'] = pd.cut(df_fe['age'], 
                                   bins=[0, 45, 60, 100], 
                                   labels=['young', 'middle', 'old'])
        
        # 创建血压分类
        df_fe['bp_category'] = pd.cut(df_fe['trestbps'], 
                                     bins=[0, 120, 140, 300], 
                                     labels=['normal', 'elevated', 'high'])
        
        # 创建胆固醇分类
        df_fe['chol_category'] = pd.cut(df_fe['chol'], 
                                       bins=[0, 200, 240, 600], 
                                       labels=['desirable', 'borderline', 'high'])
        
        # 心率储备
        df_fe['heart_rate_reserve'] = 220 - df_fe['age'] - df_fe['thalach']
        
        # 胸痛严重程度
        chest_pain_severity = {0: 0, 1: 1, 2: 2, 3: 3}
        df_fe['cp_severity'] = df_fe['cp'].map(chest_pain_severity)
        
        # 风险评分组合
        df_fe['risk_score'] = (df_fe['age'] * 0.1 + 
                              df_fe['trestbps'] * 0.01 + 
                              df_fe['chol'] * 0.001 + 
                              df_fe['oldpeak'] * 2)
        
        print(f"新增特征: {len(df_fe.columns) - len(df.columns)} 个")
        return df_fe
    
    def encode_categorical(self, df):
        """类别变量编码"""
        print("=== 类别变量编码 ===")
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            self.label_encoders[col] = le
            print(f"{col}: 编码为 {len(le.classes_)} 个类别")
        
        return df_encoded
    
    def normalize_features(self, X_train, X_val, X_test):
        """特征标准化"""
        print("=== 特征标准化 ===")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_names = X_train.columns.tolist()
        
        return (pd.DataFrame(X_train_scaled, columns=self.feature_names),
                pd.DataFrame(X_val_scaled, columns=self.feature_names),
                pd.DataFrame(X_test_scaled, columns=self.feature_names))
    
    def split_data(self, X, y, test_size=0.3, val_size=0.5, random_state=42):
        """数据分割"""
        print("=== 数据分割 ===")
        # 首先分割出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 再从剩余数据中分割出验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        print(f"训练集: {X_train.shape[0]} 样本")
        print(f"验证集: {X_val.shape[0]} 样本")
        print(f"测试集: {X_test.shape[0]} 样本")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

class EDAAnalyzer:
    """探索性数据分析类"""
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    def plot_target_distribution(self, df):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        counts = df['target'].value_counts()
        plt.pie(counts.values, labels=['无心脏病', '有心脏病'], 
                autopct='%1.1f%%', colors=self.colors[:2])
        plt.title('目标变量分布')
        plt.subplot(1, 2, 2)
        sns.countplot(data=df, x='target', palette=self.colors[:2])
        plt.title('目标变量计数')
        plt.xticks([0, 1], ['无心脏病', '有心脏病'])
        plt.tight_layout()
        plt.show()
    def plot_feature_distributions(self, df, cols_per_row=4):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        n_cols = len(numeric_cols)
        n_rows = (n_cols + cols_per_row - 1) // cols_per_row
        plt.figure(figsize=(16, 4 * n_rows))
        for i, col in enumerate(numeric_cols):
            plt.subplot(n_rows, cols_per_row, i + 1)
            sns.histplot(data=df, x=col, hue='target', kde=True, alpha=0.7)
            plt.title(f'{col} 分布')
        plt.tight_layout()
        plt.show()
    def plot_correlation_matrix(self, df):
        plt.figure(figsize=(12, 10))
        corr_matrix = df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('特征相关性矩阵')
        plt.show()
        target_corr = corr_matrix['target'].abs().sort_values(ascending=False)
        print("与目标变量相关性排序:")
        print(target_corr[1:].head(10))
    def plot_pairwise_relationships(self, df, important_features):
        plt.figure(figsize=(15, 12))
        features_to_plot = important_features + ['target']
        subset_df = df[features_to_plot]
        sns.pairplot(subset_df, hue='target', diag_kind='kde', 
                    palette=self.colors[:2])
        plt.suptitle('重要特征的两两关系', y=1.02)
        plt.show()
    def statistical_summary(self, df):
        print("=== 统计摘要 ===")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        for col in numeric_cols:
            print(f"\n{col}:")
            group_stats = df.groupby('target')[col].describe()
            print(group_stats)
            group0 = df[df['target'] == 0][col]
            group1 = df[df['target'] == 1][col]
            t_stat, p_value = stats.ttest_ind(group0, group1)
            print(f"t检验 p值: {p_value:.4f}")
