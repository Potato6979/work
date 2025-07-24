import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

# 导入自定义模块
from src.data_preprocessing import DataPreprocessor, EDAAnalyzer
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.visualization import VisualizationTools

warnings.filterwarnings('ignore')

class HeartDiseaseMLProject:
    def __init__(self, data_path='data/heart.csv', random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.preprocessor = DataPreprocessor()
        self.eda_analyzer = EDAAnalyzer()
        self.trainer = ModelTrainer(random_state=random_state)
        self.evaluator = ModelEvaluator()
        self.visualizer = VisualizationTools()
        
        # 数据存储
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.results = None
        
        print("=== 心脏病预测机器学习项目 ===")
        print(f"项目初始化完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def load_and_explore_data(self):
        """数据加载和探索性分析"""
        print("\n" + "="*50)
        print("第一步: 数据加载和探索性分析")
        print("="*50)
        
        # 加载数据
        self.raw_data = self.preprocessor.load_data(self.data_path)
        
        # 基本信息
        basic_stats = self.preprocessor.basic_info(self.raw_data)
        
        # 探索性数据分析
        print("\n进行探索性数据分析...")
        self.eda_analyzer.plot_target_distribution(self.raw_data)
        self.eda_analyzer.plot_feature_distributions(self.raw_data)
        self.eda_analyzer.plot_correlation_matrix(self.raw_data)
        
        # 统计摘要
        self.eda_analyzer.statistical_summary(self.raw_data)
        
        # 数据概览图
        self.visualizer.plot_data_overview(self.raw_data)
        
        return basic_stats
    
    def preprocess_data(self):
        """数据预处理"""
        print("\n" + "="*50)
        print("第二步: 数据预处理")
        print("="*50)
        
        # 处理缺失值
        self.processed_data = self.preprocessor.handle_missing_values(self.raw_data.copy())
        
        # 异常值检测和处理
        self.processed_data = self.preprocessor.detect_outliers(self.processed_data)
        
        # 特征工程
        self.processed_data = self.preprocessor.feature_engineering(self.processed_data)
        
        # 类别变量编码
        self.processed_data = self.preprocessor.encode_categorical(self.processed_data)
        
        # 准备特征和目标变量
        X = self.processed_data.drop('target', axis=1)
        y = self.processed_data['target']
        
        # 数据分割
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
            self.preprocessor.split_data(X, y, test_size=0.2, val_size=0.25)
        
        # 特征标准化
        self.X_train, self.X_val, self.X_test = \
            self.preprocessor.normalize_features(self.X_train, self.X_val, self.X_test)
        
        print(f"预处理完成!")
        print(f"最终特征数量: {self.X_train.shape[1]}")
        print(f"训练集大小: {self.X_train.shape[0]}")
        print(f"验证集大小: {self.X_val.shape[0]}")
        print(f"测试集大小: {self.X_test.shape[0]}")
    
    def train_models(self, tune_hyperparams=True):
        """模型训练"""
        print("\n" + "="*50)
        print("第三步: 模型训练")
        print("="*50)
        
        # 训练所有基础模型
        self.trainer.train_all_models(self.X_train, self.y_train, tune_hyperparams)
        
        # 创建集成模型
        self.trainer.create_ensemble_models(self.X_train, self.y_train)
        
        print(f"\n训练完成! 共训练了 {len(self.trainer.trained_models)} 个模型")
        
        # 绘制学习曲线（选择几个代表性模型）
        representative_models = ['logistic', 'random_forest', 'xgboost']
        for model_name in representative_models:
            if model_name in self.trainer.trained_models:
                print(f"\n绘制 {model_name} 学习曲线...")
                self.trainer.plot_learning_curves(model_name, self.X_train, self.y_train)
    
    def evaluate_models(self):
        """模型评估"""
        print("\n" + "="*50)
        print("第四步: 模型评估")
        print("="*50)
        
        # 评估所有模型
        self.results = self.evaluator.evaluate_all_models(
            self.trainer.trained_models, self.X_test, self.y_test
        )
        
        # 绘制性能对比图
        self.visualizer.plot_model_performance_comparison(self.results)
        
        # 绘制混淆矩阵
        self.evaluator.plot_confusion_matrices(self.y_test)
        
        # 绘制ROC曲线
        self.evaluator.plot_roc_curves(self.y_test)
        
        # 绘制PR曲线
        self.evaluator.plot_precision_recall_curves(self.y_test)
        
        # 特征重要性分析（对几个主要模型）
        important_models = ['random_forest', 'xgboost', 'gradient_boosting']
        for model_name in important_models:
            if model_name in self.trainer.trained_models:
                print(f"\n分析 {model_name} 特征重要性...")
                self.evaluator.feature_importance_analysis(
                    self.trainer.trained_models[model_name], 
                    model_name, 
                    self.X_test, 
                    self.y_test, 
                    self.X_train.columns.tolist()
                )
        
        # SHAP分析（选择最佳模型）
        best_model_name = self.results.iloc[0]['model']
        if best_model_name in self.trainer.trained_models:
            print(f"\n对最佳模型 {best_model_name} 进行SHAP分析...")
            self.evaluator.shap_analysis(
                self.trainer.trained_models[best_model_name],
                self.X_test,
                self.X_train.columns.tolist(),
                best_model_name
            )
        
        # 错误分析
        self.evaluator.error_analysis(
            self.y_test, self.X_test, self.X_train.columns.tolist()
        )
        
        # 生成详细报告
        self.evaluator.generate_classification_report(self.y_test)
        
        return self.results
    
    def generate_final_report(self):
        """生成最终报告"""
        print("\n" + "="*50)
        print("第五步: 生成最终报告")
        print("="*50)
        
        # 找出最佳模型
        best_model_info = self.results.iloc[0]
        
        print(f"\n=== 项目总结报告 ===")
        print(f"数据集规模: {len(self.raw_data)} 个样本")
        print(f"特征数量: {self.X_train.shape[1]} 个")
        print(f"训练模型数量: {len(self.trainer.trained_models)} 个")
        
        print(f"\n=== 最佳模型性能 ===")
        print(f"最佳模型: {best_model_info['model']}")
        print(f"准确率: {best_model_info['accuracy']:.4f}")
        print(f"精确率: {best_model_info['precision']:.4f}")  
        print(f"召回率: {best_model_info['recall']:.4f}")
        print(f"F1分数: {best_model_info['f1']:.4f}")
        if 'auc_roc' in best_model_info:
            print(f"AUC-ROC: {best_model_info['auc_roc']:.4f}")
        
        print(f"\n=== 性能排名 ===")
        for i, row in self.results.head().iterrows():
            print(f"{i+1}. {row['model']}: {row['accuracy']:.4f}")
        
        # 模型比较可视化
        self.evaluator.model_comparison_plot()
        
        print(f"\n项目完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def save_results(self, output_dir='./output'):
        """保存结果"""
        print("\n" + "="*50)
        print("保存结果")
        print("="*50)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(output_dir, 'trained_models.pkl')
        self.trainer.save_models(model_path)
        
        # 保存评估结果
        results_path = os.path.join(output_dir, 'evaluation_results.pkl')
        self.evaluator.save_results(results_path)
        
        # 保存预处理器
        import joblib
        preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
        joblib.dump(self.preprocessor, preprocessor_path)
        
        # 保存结果CSV
        csv_path = os.path.join(output_dir, 'model_results.csv')
        self.results.to_csv(csv_path, index=False)
        
        print(f"所有结果已保存到: {output_dir}")
    
    def run_complete_pipeline(self, save_results=True):
        """运行完整的机器学习流程"""
        try:
            # 1. 数据加载和探索
            self.load_and_explore_data()
            
            # 2. 数据预处理
            self.preprocess_data()
            
            # 3. 模型训练
            self.train_models(tune_hyperparams=True)
            
            # 4. 模型评估
            self.evaluate_models()
            
            # 5. 生成报告
            self.generate_final_report()
            
            # 6. 保存结果
            if save_results:
                self.save_results()
            
            print("\n🎉 项目执行完成!")
            
        except Exception as e:
            print(f"\n❌ 项目执行出错: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    # 检查数据文件是否存在
    data_path = 'data/heart.csv'
    if not os.path.exists(data_path):
        print(f"数据文件 {data_path} 不存在!")
        print("请从以下链接下载数据:")
        print("https://archive.ics.uci.edu/ml/datasets/heart+disease")
        return
    
    # 创建项目实例
    project = HeartDiseaseMLProject(data_path=data_path, random_state=42)
    
    # 运行完整流程
    project.run_complete_pipeline(save_results=True)
    
    # 创建交互式仪表板（可选）
    try:
        project.visualizer.create_interactive_dashboard(
            project.processed_data, project.results
        )
    except Exception as e:
        print(f"交互式仪表板创建失败: {e}")

if __name__ == "__main__":
    main()