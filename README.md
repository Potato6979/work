# 基于机器学习的心脏病预测分析项目

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 项目简介

本项目使用多种机器学习算法对心脏病进行预测分析，包括数据预处理、特征工程、模型训练、性能评估和结果可视化等完整流程。

## 主要特性

- **完整的数据科学流程**: 从数据加载到模型部署的端到端实现
- **多算法对比**: 8种经典机器学习算法 + 3种集成方法
- **可解释性分析**: SHAP值分析和特征重要性排序
- **丰富的可视化**: 20+种专业图表展示分析结果
- **高度模块化**: 易于扩展和维护的代码结构

## 安装指南

### 1. 克隆仓库
```bash
git clone https://github.com/[your-username]/heart-disease-prediction-ml.git
cd heart-disease-prediction-ml
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或者
venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

## 快速开始

### 数据准备
下载UCI心脏病数据集并放置在 `data/` 目录下:
```bash
mkdir data
# 将heart.csv文件放入data目录
```

### 运行完整流程
```bash
python main.py
```

### Jupyter Notebook演示
```bash
jupyter notebook
# 打开notebooks/目录下的演示文件
```

## 项目结构
```
heart-disease-prediction-ml/
├── data/                   # 数据文件
├── src/                    # 源代码
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── visualization.py
├── notebooks/              # Jupyter notebooks
├── output/                 # 输出结果
├── tests/                  # 测试文件
├── main.py                # 主程序
├── requirements.txt       # 依赖包
└── README.md             # 项目说明
```

## 核心功能

### 1. 数据预处理
- 缺失值处理
- 异常值检测
- 特征工程
- 数据标准化

### 2. 模型训练
- 8种基础算法
- 超参数自动调优
- 集成学习方法
- 交叉验证

### 3. 模型评估
- 多维度性能指标
- ROC/PR曲线分析
- 混淆矩阵可视化
- 特征重要性分析

### 4. 结果可视化
- 数据探索图表
- 模型性能对比
- 特征关系分析
- 交互式仪表板

## 实验结果

### 最佳模型性能
- **算法**: 软投票集成
- **准确率**: 90.4%
- **AUC-ROC**: 0.954
- **F1分数**: 90.3%

### 重要发现
1. 主要血管数(ca)是最重要的预测因子
2. 胸痛类型(cp)和地中海贫血(thal)同样重要
3. 集成方法显著提升了预测性能

## 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境设置
```bash
pip install -e ".[dev]"
pre-commit install
```

### 代码风格
本项目使用Black进行代码格式化：
```bash
black src/ tests/
```

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- UCI机器学习库提供数据集
- scikit-learn团队提供优秀的ML库
- 所有开源贡献者

## 联系方式

- 作者: [Your Name]
- 邮箱: [your.email@example.com]
- GitHub: [@your-username](https://github.com/your-username) 