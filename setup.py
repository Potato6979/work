from setuptools import setup, find_packages

setup(
    name="heart-disease-prediction-ml",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="基于机器学习的心脏病预测分析项目",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/[your-username]/heart-disease-prediction-ml",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "xgboost>=1.5.0",
        "shap>=0.40.0",
        "plotly>=5.0.0",
        "jupyter>=1.0.0",
        "joblib>=1.1.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
    },
) 