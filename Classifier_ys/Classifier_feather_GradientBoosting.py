# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.feature_selection import SelectPercentile, chi2, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, average_precision_score
import shap
import joblib
import time
import pickle
import matplotlib.pyplot as plt
import config_train


from FeatureEngineering_success import extract_features


features_number = [{"AAC": 20}, {"CKSAAP": 2400}, {"CTDC": 39}, {"CTDT": 39}, {"CTDD": 195}, {"CTriad": 343}, {"DPC": 400}, {"GAAC": 5}, {"GDPC": 25},
                   {"CKSAAGP":150}, {"DDE": 400}, {"GTPC": 125}, {"KSCTriad": 343}, {"TPC": 8000}]  # ys:
# feature_ranges = {"AAC": (0, 20), "CKSAAP": (20, 1620), "CTDC": (1620, 1659), "CTDT": (1659, 1698), "CTDD": (1698, 1893),
#                   "CTriad": (1893, 2236), "DPC": (2236, 2636), "GAAC": (2636, 2641), "GDPC": (2641, 2666),
#                   'CKSAAGP': (2666, 2816), 'DDE': (2816, 3216), 'GTPC': (3216, 3341), 'KSCTriad': (3341, 3684), 'TPC': (3684, 11684)}
feature_ranges = {"AAC": (0, 20), "CKSAAP": (20, 2420), "CTDC": (2420, 2459), "CTDT": (2459, 2498), "CTDD": (2498, 2693),
                  "CTriad": (2693, 3036), "DPC": (3036, 3436), "GAAC": (3436, 3441), "GDPC": (3441, 3466),
                  "CKSAAGP": (3466, 3616), "DDE": (3616, 4016), "GTPC": (4016, 4141), "KSCTriad": (4141, 4484), "TPC": (4484, 12484)}
feather_type_num = ['', 'AAC/', 'CKSAAGP/', 'CKSAAP/', 'CTDC/', 'CTDD/', 'CTDT/', 'CTriad/', 'DDE/', 'DPC/', 'GAAC/', 'GDPC/', 'GTPC/', 'KSCTriad/', 'TPC/']
normallization_type = ['smote']   # 'all', 'fiveFold', 'MinMaxScaler', 'StandardScaler', 'smote'
# normallization_type = ['Chi2', 'f_classif', 'Mutual Information', 'Variance Threshold']
# for i in range(0, len(feather_type_num)):
for normal_type in normallization_type:
    i=10    # 0
    feather_type = f"{feather_type_num[i]}"   # AAC, CKSAAGP, CKSAAP, CTDC, CTDD, CTDT, CTriad, DDE, DPC, GAAC, GDPC, GTPC, KSCTriad, TPC
    datasets = pd.read_csv(f"../features_ys/{feather_type}train.csv", low_memory=False)     # , header=None
    X_train_src = pd.read_csv(f"../features_ys/{feather_type}train_src.csv", low_memory=False)     # , header=None
    # datasets.insert(0, 'label', pd.Series(np.concatenate([np.zeros(5112), np.ones(7462)])))
    datasets.insert(0, 'label', pd.Series(np.concatenate([np.zeros(3721), np.ones(2358)])))
    # X_train, y_train = np.array(datasets.iloc[:, 1:]), np.array(datasets.iloc[:, 0])
    X_train, y_train = datasets.iloc[:, 1:], datasets.iloc[:, 0]

    X_test = pd.read_csv(f"../features_ys/{feather_type}test.csv")
    X_test_src = pd.read_csv(f"../features_ys/{feather_type}test_src.csv")
    y_test = pd.Series(np.concatenate([np.zeros(len(X_test) // 2), np.ones(len(X_test) - len(X_test) // 2)]))
    X, y = X_train, y_train

    feather_type = f"{normal_type}"
    if normal_type not in ['MinMaxScaler', 'StandardScaler']: # smote
        X_scaled = X

    X_selected = X_scaled

    # 数据增强，处理不平衡数据
    smote = SMOTE(random_state=42)

    # # 划分数据集
    X_train, y_train = X_selected, y
    X_train_encoding, X_test_encoding = X_train, X_test

    # 定义评估函数
    def evaluate_model(y_test, y_pred, y_prob):
        # 计算各项指标
        recall = recall_score(y_test, y_pred)   # , average='macro'
        precision = precision_score(y_test, y_pred)     # , average='macro'
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)   # , average='macro'
        mcc = matthews_corrcoef(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        auroc = roc_auc_score(y_test, y_prob)   # , multi_class='ovr'
        auprc = average_precision_score(y_test, y_prob)

        res = {'Recall': recall, 'Precision': precision, 'Accuracy': accuracy, 'F1 Score': f1, 'MCC': mcc,
               'Specificity': specificity, 'AUROC': auroc, 'AUPRC': auprc
               }

        return res


    # 模型定义及其超参数
    models = [
        ('GradientBoosting', GradientBoostingClassifier(random_state=42), {'learning_rate': [0.2], 'n_estimators': [200]}),
    ]
    # 定义模型及其超参数网格

    metric_res = {}         # 将 metric 值存储到字典中
    shap_values_dict = {}   # 将 SHAP 值存储到字典中
    feature_importance_dict = {}

    # 特征提取方法列表
    methods = ["HashingVectorizer"]

    # 遍历特征工程(特征提取)方法和模型
    for method in methods:

        # 将训练集和测试集的序列组合
        corpus = pd.concat([X_train_src, X_test_src], axis=0).iloc[:, 0].tolist()
        X_features = extract_features(method, corpus)

        # # 确保特征提取后是 pandas DataFrame 格式
        X_train = np.concatenate((X_features[:len(X_train_src)], X_train_encoding), axis=1)
        X_test = np.concatenate((X_features[len(X_train_src):], X_test_encoding), axis=1)


        # 特征选择步骤
        # 选择前 50% 的特征
        if normallization_type == 'Chi2':
            selector_chi2 = SelectPercentile(score_func=chi2, percentile=50)
            X_train = selector_chi2.fit_transform(X_train, y)

        # 评估并优化每个模型
        for name, model, params in models:
            grid_search = GridSearchCV(model, params, cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42), scoring='accuracy', n_jobs=-1) # , n_jobs=-1, cv=5, scoring='f1'

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            y_pred = best_model.predict(X_test)
            if hasattr(best_model, 'predict_proba'):
                y_prob = best_model.predict_proba(X_test)[:, 1]

            metric_res[name+'_'+method] = evaluate_model(y_test, y_pred, y_prob)





