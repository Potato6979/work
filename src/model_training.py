import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             VotingClassifier, StackingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (GridSearchCV, cross_val_score, 
                                   validation_curve, learning_curve)
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = self._initialize_models()
        self.trained_models = {}
        self.best_params = {}
        self.cv_scores = {}
    
    def _initialize_models(self):
        """初始化所有模型"""
        return {
            'logistic': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'svm': SVC(
                probability=True, 
                random_state=self.random_state
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.random_state
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'knn': KNeighborsClassifier(),
            'naive_bayes': GaussianNB(),
            'mlp': MLPClassifier(
                random_state=self.random_state,
                max_iter=1000
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.random_state
            )
        }
    
    def get_param_grids(self):
        """获取超参数搜索空间"""
        return {
            'logistic': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 10],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            'decision_tree': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        }
    
    def hyperparameter_tuning(self, model_name, X_train, y_train, cv=5, 
                            scoring='accuracy', n_jobs=-1):
        """超参数调优"""
        print(f"开始调优 {model_name}...")
        
        param_grids = self.get_param_grids()
        if model_name not in param_grids:
            print(f"没有为 {model_name} 定义参数网格")
            return self.models[model_name]
        
        grid_search = GridSearchCV(
            estimator=self.models[model_name],
            param_grid=param_grids[model_name],
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params[model_name] = grid_search.best_params_
        print(f"{model_name} 最佳参数: {grid_search.best_params_}")
        print(f"{model_name} 最佳得分: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def cross_validation(self, model, X_train, y_train, cv=5, scoring='accuracy'):
        """交叉验证"""
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        return scores
    
    def train_all_models(self, X_train, y_train, tune_hyperparams=True):
        """训练所有模型"""
        print("=== 开始训练所有模型 ===")
        
        for name in self.models.keys():
            print(f"\n训练 {name}...")
            
            if tune_hyperparams:
                # 超参数调优
                model = self.hyperparameter_tuning(name, X_train, y_train)
            else:
                model = self.models[name]
                model.fit(X_train, y_train)
            
            # 交叉验证
            cv_scores = self.cross_validation(model, X_train, y_train)
            self.cv_scores[name] = cv_scores
            
            # 保存训练好的模型
            self.trained_models[name] = model
            
            print(f"{name} 交叉验证得分: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def create_ensemble_models(self, X_train, y_train):
        """创建集成模型"""
        print("\n=== 创建集成模型 ===")
        
        # 选择表现最好的几个模型进行集成
        top_models = sorted(self.cv_scores.items(), 
                           key=lambda x: x[1].mean(), reverse=True)[:5]
        
        estimators = []
        for name, _ in top_models:
            estimators.append((name, self.trained_models[name]))
        
        # 硬投票分类器
        hard_voting = VotingClassifier(
            estimators=estimators, 
            voting='hard'
        )
        hard_voting.fit(X_train, y_train)
        self.trained_models['hard_voting'] = hard_voting
        
        # 软投票分类器
        soft_voting = VotingClassifier(
            estimators=estimators, 
            voting='soft'
        )
        soft_voting.fit(X_train, y_train)
        self.trained_models['soft_voting'] = soft_voting
        
        # Stacking分类器
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=self.random_state),
            cv=5
        )
        stacking.fit(X_train, y_train)
        self.trained_models['stacking'] = stacking
        
        print("集成模型创建完成")
    
    def plot_validation_curves(self, model_name, X_train, y_train, param_name, param_range):
        """绘制验证曲线"""
        train_scores, validation_scores = validation_curve(
            self.models[model_name], X_train, y_train,
            param_name=param_name, param_range=param_range,
            cv=5, scoring='accuracy'
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_scores.mean(axis=1), 'o-', 
                label='训练得分', color='blue')
        plt.fill_between(param_range, 
                        train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1),
                        alpha=0.1, color='blue')
        
        plt.plot(param_range, validation_scores.mean(axis=1), 'o-', 
                label='验证得分', color='red')
        plt.fill_between(param_range, 
                        validation_scores.mean(axis=1) - validation_scores.std(axis=1),
                        validation_scores.mean(axis=1) + validation_scores.std(axis=1),
                        alpha=0.1, color='red')
        
        plt.xlabel(param_name)
        plt.ylabel('准确率')
        plt.title(f'{model_name} 验证曲线')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_learning_curves(self, model_name, X_train, y_train):
        """绘制学习曲线"""
        train_sizes, train_scores, validation_scores = learning_curve(
            self.trained_models[model_name], X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring='accuracy'
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', 
                label='训练得分', color='blue')
        plt.fill_between(train_sizes, 
                        train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1),
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes, validation_scores.mean(axis=1), 'o-', 
                label='验证得分', color='red')
        plt.fill_between(train_sizes, 
                        validation_scores.mean(axis=1) - validation_scores.std(axis=1),
                        validation_scores.mean(axis=1) + validation_scores.std(axis=1),
                        alpha=0.1, color='red')
        
        plt.xlabel('训练样本数')
        plt.ylabel('准确率')
        plt.title(f'{model_name} 学习曲线')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save_models(self, filepath):
        """保存模型"""
        joblib.dump({
            'models': self.trained_models,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_models(self, filepath):
        """加载模型"""
        data = joblib.load(filepath)
        self.trained_models = data['models']
        self.best_params = data['best_params']
        self.cv_scores = data['cv_scores']
        print(f"模型已从 {filepath} 加载") 