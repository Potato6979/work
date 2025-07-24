import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve, precision_recall_curve,
                           confusion_matrix, classification_report, average_precision_score)
from sklearn.inspection import permutation_importance
import shap
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        self.predictions = {}
        self.probabilities = {}
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        """评估单个模型"""
        # 预测
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        # 保存预测结果
        self.predictions[model_name] = y_pred
        if y_prob is not None:
            self.probabilities[model_name] = y_prob
        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'specificity': self._calculate_specificity(y_test, y_pred),
            'sensitivity': recall_score(y_test, y_pred)  # 同recall
        }
        if y_prob is not None:
            metrics['auc_roc'] = roc_auc_score(y_test, y_prob)
            metrics['auc_pr'] = average_precision_score(y_test, y_prob)
        self.results[model_name] = metrics
        return metrics
    
    def _calculate_specificity(self, y_true, y_pred):
        """计算特异性"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)
    
    def evaluate_all_models(self, models, X_test, y_test):
        """评估所有模型"""
        print("=== 模型评估结果 ===")
        results_df = []
        for name, model in models.items():
            print(f"\n评估 {name}...")
            metrics = self.evaluate_model(model, name, X_test, y_test)
            metrics['model'] = name
            results_df.append(metrics)
        results_df = pd.DataFrame(results_df)
        results_df = results_df.sort_values('accuracy', ascending=False)
        print("\n=== 所有模型性能对比 ===")
        print(results_df.round(4))
        return results_df
    
    def plot_confusion_matrices(self, y_test, models_to_plot=None):
        """绘制混淆矩阵"""
        if models_to_plot is None:
            models_to_plot = list(self.predictions.keys())[:6]  # 只显示前6个
        n_models = len(models_to_plot)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        for i, model_name in enumerate(models_to_plot):
            if i >= len(axes):
                break
            cm = confusion_matrix(y_test, self.predictions[model_name])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[i], cbar=False)
            axes[i].set_title(f'{model_name}')
            axes[i].set_xlabel('预测值')
            axes[i].set_ylabel('真实值')
        # 隐藏多余的子图
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, y_test, models_to_plot=None):
        """绘制ROC曲线"""
        if models_to_plot is None:
            models_to_plot = list(self.probabilities.keys())
        plt.figure(figsize=(10, 8))
        for model_name in models_to_plot:
            if model_name in self.probabilities:
                fpr, tpr, _ = roc_curve(y_test, self.probabilities[model_name])
                auc = roc_auc_score(y_test, self.probabilities[model_name])
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('ROC曲线对比')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_precision_recall_curves(self, y_test, models_to_plot=None):
        """绘制Precision-Recall曲线"""
        if models_to_plot is None:
            models_to_plot = list(self.probabilities.keys())
        plt.figure(figsize=(10, 8))
        for model_name in models_to_plot:
            if model_name in self.probabilities:
                precision, recall, _ = precision_recall_curve(y_test, self.probabilities[model_name])
                ap = average_precision_score(y_test, self.probabilities[model_name])
                plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})')
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('Precision-Recall曲线对比')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def feature_importance_analysis(self, model, model_name, X_test, y_test, feature_names):
        """特征重要性分析"""
        print(f"\n=== {model_name} 特征重要性分析 ===")
        # 树模型的内置特征重要性
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df.head(10), x='importance', y='feature')
            plt.title(f'{model_name} - 特征重要性（内置）')
            plt.xlabel('重要性')
            plt.show()
            print("Top 10 重要特征:")
            print(importance_df.head(10))
        # 排列重要性
        try:
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42
            )
            perm_df = pd.DataFrame({
                'feature': feature_names,
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('importance', ascending=False)
            plt.figure(figsize=(10, 8))
            plt.errorbar(perm_df.head(10)['importance'], 
                        range(len(perm_df.head(10))), 
                        xerr=perm_df.head(10)['std'], fmt='o')
            plt.yticks(range(len(perm_df.head(10))), perm_df.head(10)['feature'])
            plt.xlabel('排列重要性')
            plt.title(f'{model_name} - 排列重要性')
            plt.gca().invert_yaxis()
            plt.grid(True)
            plt.show()
            print("\n排列重要性 Top 10:")
            print(perm_df.head(10))
        except Exception as e:
            print(f"排列重要性计算失败: {e}")
    
    def shap_analysis(self, model, X_test, feature_names, model_name, max_display=10):
        """SHAP值分析"""
        try:
            print(f"\n=== {model_name} SHAP分析 ===")
            # 选择合适的Explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model, X_test)
            else:
                explainer = shap.LinearExplainer(model, X_test)
            shap_values = explainer(X_test[:100])  # 只分析前100个样本
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test[:100], 
                            feature_names=feature_names, show=False)
            plt.title(f'{model_name} - SHAP Summary Plot')
            plt.tight_layout()
            plt.show()
            # Feature importance plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test[:100], 
                            feature_names=feature_names, 
                            plot_type="bar", show=False)
            plt.title(f'{model_name} - SHAP Feature Importance')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"SHAP分析失败: {e}")
    
    def cross_validation_analysis(self, models, X, y, cv=5):
        """交叉验证分析"""
        print("=== 交叉验证分析 ===")
        cv_results = {}
        for name, model in models.items():
            try:
                # 交叉验证预测
                y_pred_cv = cross_val_predict(model, X, y, cv=cv)
                # 计算交叉验证指标
                cv_metrics = {
                    'accuracy': accuracy_score(y, y_pred_cv),
                    'precision': precision_score(y, y_pred_cv),
                    'recall': recall_score(y, y_pred_cv),
                    'f1': f1_score(y, y_pred_cv)
                }
                cv_results[name] = cv_metrics
                print(f"{name}: Accuracy={cv_metrics['accuracy']:.4f}, F1={cv_metrics['f1']:.4f}")
            except Exception as e:
                print(f"{name} 交叉验证失败: {e}")
        return cv_results
    
    def error_analysis(self, y_test, X_test, feature_names, models_to_analyze=None):
        """错误分析"""
        if models_to_analyze is None:
            models_to_analyze = list(self.predictions.keys())[:3]  # 分析前3个模型
        print("=== 错误分析 ===")
        for model_name in models_to_analyze:
            if model_name not in self.predictions:
                continue
            y_pred = self.predictions[model_name]
            # 找出错误分类的样本
            errors = y_test != y_pred
            error_indices = np.where(errors)[0]
            print(f"\n{model_name} 错误分析:")
            print(f"错误样本数: {len(error_indices)}")
            print(f"错误率: {len(error_indices)/len(y_test):.4f}")
            if len(error_indices) > 0:
                # 分析假阳性和假阴性
                false_positives = np.where((y_test == 0) & (y_pred == 1))[0]
                false_negatives = np.where((y_test == 1) & (y_pred == 0))[0]
                print(f"假阳性: {len(false_positives)}")
                print(f"假阴性: {len(false_negatives)}")
                # 分析错误样本的特征分布
                if len(error_indices) > 5:  # 只有足够的错误样本才进行分析
                    error_samples = X_test.iloc[error_indices]
                    correct_samples = X_test.iloc[~errors]
                    print(f"\n错误样本特征统计 (前5个特征):")
                    for feature in feature_names[:5]:
                        if feature in error_samples.columns:
                            error_mean = error_samples[feature].mean()
                            correct_mean = correct_samples[feature].mean()
                            print(f"{feature}: 错误样本均值={error_mean:.3f}, 正确样本均值={correct_mean:.3f}")
    
    def generate_classification_report(self, y_test, model_name=None):
        """生成详细分类报告"""
        if model_name is None:
            # 选择表现最好的模型
            best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        else:
            best_model = model_name
        if best_model not in self.predictions:
            print(f"模型 {best_model} 的预测结果不存在")
            return
        y_pred = self.predictions[best_model]
        print(f"\n=== {best_model} 详细分类报告 ===")
        print(classification_report(y_test, y_pred, 
                                  target_names=['无心脏病', '有心脏病']))
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n混淆矩阵:")
        print(f"真实\\预测    无心脏病    有心脏病")
        print(f"无心脏病        {cm[0,0]}         {cm[0,1]}")
        print(f"有心脏病        {cm[1,0]}         {cm[1,1]}")
        # 计算其他指标
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)  # 敏感性
        specificity = tn / (tn + fp)  # 特异性
        ppv = tp / (tp + fp)  # 阳性预测值
        npv = tn / (tn + fn)  # 阴性预测值
        print(f"\n临床指标:")
        print(f"敏感性 (Sensitivity): {sensitivity:.4f}")
        print(f"特异性 (Specificity): {specificity:.4f}")
        print(f"阳性预测值 (PPV): {ppv:.4f}")
        print(f"阴性预测值 (NPV): {npv:.4f}")
    
    def model_comparison_plot(self):
        """模型性能对比图"""
        if not self.results:
            print("没有评估结果可供比较")
            return
        # 准备数据
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        for i, metric in enumerate(metrics):
            values = [self.results[model].get(metric, 0) for model in models]
            bars = axes[i].bar(models, values, alpha=0.7)
            axes[i].set_title(f'{metric.upper()} 对比')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            # 添加数值标签
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.show()
    
    def save_results(self, filepath):
        """保存评估结果"""
        results_data = {
            'metrics': self.results,
            'predictions': self.predictions,
            'probabilities': self.probabilities
        }
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(results_data, f)
        print(f"评估结果已保存到: {filepath}") 