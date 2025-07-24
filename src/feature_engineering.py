import pandas as pd

def add_feature_engineering(df):
    """
    独立特征工程函数，可在主流程或notebook中单独调用。
    """
    # 年龄分组
    df['age_group'] = pd.cut(df['age'], bins=[0, 45, 60, 100], labels=['young', 'middle', 'old'])
    # 血压分类
    df['bp_category'] = pd.cut(df['trestbps'], bins=[0, 120, 140, 300], labels=['normal', 'elevated', 'high'])
    # 胆固醇分类
    df['chol_category'] = pd.cut(df['chol'], bins=[0, 200, 240, 600], labels=['desirable', 'borderline', 'high'])
    # 心率储备
    df['heart_rate_reserve'] = 220 - df['age'] - df['thalach']
    # 胸痛严重程度
    chest_pain_severity = {0: 0, 1: 1, 2: 2, 3: 3}
    df['cp_severity'] = df['cp'].map(chest_pain_severity)
    # 风险评分组合
    df['risk_score'] = (df['age'] * 0.1 + df['trestbps'] * 0.01 + df['chol'] * 0.001 + df['oldpeak'] * 2)
    return df

# 可根据需要扩展更多特征工程函数
