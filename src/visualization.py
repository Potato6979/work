# ... 见heart_disease_ml_project.md可视化模块，完整实现VisualizationTools类 ... 
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class VisualizationTools:
    def __init__(self):
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                      '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
    
    def plot_data_overview(self, df):
        """数据概览图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 目标变量分布
        target_counts = df['target'].value_counts()
        axes[0, 0].pie(target_counts.values, labels=['无心脏病', '有心脏病'], 
                      autopct='%1.1f%%', colors=self.colors[:2])
        axes[0, 0].set_title('目标变量分布', fontsize=14, fontweight='bold')
        
        # 2. 年龄分布
        axes[0, 1].hist([df[df['target']==0]['age'], df[df['target']==1]['age']], 
                       bins=15, alpha=0.7, label=['无心脏病', '有心脏病'], 
                       color=self.colors[:2])
        axes[0, 1].set_xlabel('年龄')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('年龄分布对比')
        axes[0, 1].legend()
        
        # 3. 性别与心脏病关系
        gender_target = pd.crosstab(df['sex'], df['target'])
        gender_target.plot(kind='bar', ax=axes[0, 2], color=self.colors[:2])
        axes[0, 2].set_title('性别与心脏病关系')
        axes[0, 2].set_xlabel('性别 (0:女性, 1:男性)')
        axes[0, 2].set_ylabel('人数')
        axes[0, 2].legend(['无心脏病', '有心脏病'])
        axes[0, 2].tick_params(axis='x', rotation=0)
        
        # 4. 胸痛类型分布
        cp_target = pd.crosstab(df['cp'], df['target'])
        cp_target.plot(kind='bar', ax=axes[1, 0], color=self.colors[:2])
        axes[1, 0].set_title('胸痛类型与心脏病关系')
        axes[1, 0].set_xlabel('胸痛类型')
        axes[1, 0].set_ylabel('人数')
        axes[1, 0].legend(['无心脏病', '有心脏病'])
        axes[1, 0].tick_params(axis='x', rotation=0)
        
        # 5. 最大心率分布
        axes[1, 1].boxplot([df[df['target']==0]['thalach'], 
                           df[df['target']==1]['thalach']], 
                          labels=['无心脏病', '有心脏病'])
        axes[1, 1].set_title('最大心率分布对比')
        axes[1, 1].set_ylabel('最大心率')
        
        # 6. 胆固醇水平分布
        axes[1, 2].boxplot([df[df['target']==0]['chol'], 
                           df[df['target']==1]['chol']], 
                          labels=['无心脏病', '有心脏病'])
        axes[1, 2].set_title('胆固醇水平分布对比')
        axes[1, 2].set_ylabel('胆固醇 (mg/dl)')
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, df, figsize=(12, 10)):
        """增强版相关性热力图"""
        plt.figure(figsize=figsize)
        
        # 计算相关性矩阵
        corr_matrix = df.corr()
        
        # 创建掩码
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # 自定义颜色映射
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        # 绘制热力图
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   annot=True, fmt='.2f', annot_kws={'size': 8})
        
        plt.title('特征相关性矩阵', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        # 输出强相关特征对
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append((corr_matrix.columns[i], 
                                      corr_matrix.columns[j], 
                                      corr_val))
        
        if strong_corr:
            print("强相关特征对 (|r| > 0.5):")
            for feat1, feat2, corr_val in strong_corr:
                print(f"{feat1} - {feat2}: r = {corr_val:.3f}")
    
    def plot_feature_distributions(self, df, cols_per_row=4):
        """特征分布图（增强版）"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        n_cols = len(numeric_cols)
        n_rows = (n_cols + cols_per_row - 1) // cols_per_row
        
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(20, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(numeric_cols):
            row = i // cols_per_row
            col_idx = i % cols_per_row
            
            # 分别绘制两个类别的分布
            for target in [0, 1]:
                data = df[df['target'] == target][col]
                axes[row, col_idx].hist(data, alpha=0.7, bins=20, 
                                       label=f'Target {target}', 
                                       color=self.colors[target])
            
            axes[row, col_idx].set_title(col, fontweight='bold')
            axes[row, col_idx].legend()
            axes[row, col_idx].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_cols, n_rows * cols_per_row):
            row = i // cols_per_row
            col_idx = i % cols_per_row
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_performance_comparison(self, results_df):
        """模型性能对比图（增强版）"""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(available_metrics):
            if i >= len(axes):
                break
            
            # 排序
            sorted_data = results_df.sort_values(metric, ascending=False)
            
            bars = axes[i].bar(range(len(sorted_data)), sorted_data[metric], 
                              color=self.colors[:len(sorted_data)], alpha=0.8)
            
            axes[i].set_title(f'{metric.upper()} 对比', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_xticks(range(len(sorted_data)))
            axes[i].set_xticklabels(sorted_data['model'], rotation=45, ha='right')
            
            # 添加数值标签
            for j, (bar, value) in enumerate(zip(bars, sorted_data[metric])):
                axes[i].text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + 0.005,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 1.05)
        
        # 隐藏多余的子图
        for j in range(len(available_metrics), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_comparison(self, y_test, probabilities, models_to_plot=None):
        """ROC曲线对比图（增强版）"""
        if models_to_plot is None:
            models_to_plot = list(probabilities.keys())
        
        plt.figure(figsize=(12, 10))
        
        for i, model_name in enumerate(models_to_plot):
            if model_name in probabilities:
                from sklearn.metrics import roc_curve, roc_auc_score
                
                fpr, tpr, _ = roc_curve(y_test, probabilities[model_name])
                auc = roc_auc_score(y_test, probabilities[model_name])
                
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'{model_name} (AUC = {auc:.3f})',
                        color=self.colors[i % len(self.colors)])
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机分类器 (AUC = 0.5)')
        plt.xlabel('假阳性率 (1-特异性)', fontsize=12)
        plt.ylabel('真阳性率 (敏感性)', fontsize=12)
        plt.title('ROC曲线对比', fontsize=16, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        # 添加最优点标记
        plt.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
        plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance_comparison(self, importance_data, top_n=10):
        """特征重要性对比图"""
        if not importance_data:
            print("没有特征重要性数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, (model_name, importance_df) in enumerate(importance_data.items()):
            if i >= len(axes):
                break
            
            top_features = importance_df.head(top_n)
            
            bars = axes[i].barh(range(len(top_features)), top_features['importance'],
                               color=self.colors[i % len(self.colors)], alpha=0.8)
            
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features['feature'])
            axes[i].set_xlabel('重要性')
            axes[i].set_title(f'{model_name} - Top {top_n} 重要特征', 
                             fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            
            # 添加数值标签
            for j, (bar, value) in enumerate(zip(bars, top_features['importance'])):
                axes[i].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                           f'{value:.3f}', va='center', fontsize=9)
            
            axes[i].invert_yaxis()
        
        # 隐藏多余的子图
        for j in range(len(importance_data), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curves_comparison(self, learning_curves_data):
        """学习曲线对比"""
        if not learning_curves_data:
            print("没有学习曲线数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, (model_name, (train_sizes, train_scores, val_scores)) in enumerate(learning_curves_data.items()):
            if i >= len(axes):
                break
            
            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            val_mean = val_scores.mean(axis=1)
            val_std = val_scores.std(axis=1)
            
            axes[i].plot(train_sizes, train_mean, 'o-', color=self.colors[0], 
                        label='训练得分')
            axes[i].fill_between(train_sizes, train_mean - train_std,
                               train_mean + train_std, alpha=0.1, color=self.colors[0])
            
            axes[i].plot(train_sizes, val_mean, 'o-', color=self.colors[1], 
                        label='验证得分')
            axes[i].fill_between(train_sizes, val_mean - val_std,
                               val_mean + val_std, alpha=0.1, color=self.colors[1])
            
            axes[i].set_xlabel('训练样本数')
            axes[i].set_ylabel('准确率')
            axes[i].set_title(f'{model_name} 学习曲线', fontweight='bold')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for j in range(len(learning_curves_data), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix_heatmap(self, confusion_matrices, model_names):
        """混淆矩阵热力图"""
        n_models = len(confusion_matrices)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (cm, model_name) in enumerate(zip(confusion_matrices, model_names)):
            row = i // cols
            col = i % cols
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[row, col], cbar=False,
                       xticklabels=['预测:无', '预测:有'],
                       yticklabels=['实际:无', '实际:有'])
            
            axes[row, col].set_title(f'{model_name}', fontweight='bold')
            
            # 计算准确率
            accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
            axes[row, col].text(0.5, -0.1, f'准确率: {accuracy:.3f}', 
                               transform=axes[row, col].transAxes, 
                               ha='center', fontsize=10)
        
        # 隐藏多余的子图
        for i in range(len(confusion_matrices), rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard(self, df, results_df):
        """创建交互式仪表板"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('目标变量分布', '年龄vs最大心率', 
                              '模型性能对比', '特征相关性'),
                specs=[[{"type": "pie"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "heatmap"}]]
            )
            
            # 1. 目标变量分布饼图
            target_counts = df['target'].value_counts()
            fig.add_trace(
                go.Pie(labels=['无心脏病', '有心脏病'], 
                      values=target_counts.values,
                      name="目标分布"),
                row=1, col=1
            )
            
            # 2. 年龄vs最大心率散点图
            colors = df['target'].map({0: 'blue', 1: 'red'})
            fig.add_trace(
                go.Scatter(x=df['age'], y=df['thalach'],
                          mode='markers',
                          marker=dict(color=colors, size=8),
                          name="年龄vs心率"),
                row=1, col=2
            )
            
            # 3. 模型性能对比
            fig.add_trace(
                go.Bar(x=results_df['model'], y=results_df['accuracy'],
                      name="准确率"),
                row=2, col=1
            )
            
            # 4. 相关性热力图
            corr_matrix = df.select_dtypes(include=[np.number]).corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values,
                          x=corr_matrix.columns,
                          y=corr_matrix.columns,
                          colorscale='RdBu',
                          name="相关性"),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=False,
                            title_text="心脏病预测分析仪表板")
            fig.show()
            
        except ImportError:
            print("需要安装plotly: pip install plotly")
    
    def save_all_plots(self, save_dir='./plots'):
        """保存所有图表"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        print(f"图表已保存到: {save_dir}")