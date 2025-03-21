import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

# 设置中文字体（推荐使用系统内置字体，如 SimHei、Microsoft YaHei）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常问题

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_indices = None
        self.tree = None

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        return 1 - np.sum(prob ** 2)

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature in self.feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                if np.sum(left_idx) > 0 and np.sum(~left_idx) > 0:
                    gini = (len(y[left_idx]) * self._gini(y[left_idx]) +
                            len(y[~left_idx]) * self._gini(y[~left_idx])) / len(y)
                    if gini < best_gini:
                        best_gini = gini
                        best_feature = feature
                        best_threshold = threshold
        return best_feature, best_threshold, best_gini

    def _build_tree(self, X, y, depth=0):
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split or
                len(np.unique(y)) == 1):
            return {'class': np.argmax(np.bincount(y))}

        feature, threshold, gini = self._best_split(X, y)
        if feature is None:
            return {'class': np.argmax(np.bincount(y))}

        left_idx = X[:, feature] <= threshold
        return {
            'feature': feature,
            'threshold': threshold,
            'gini': gini,
            'left': self._build_tree(X[left_idx], y[left_idx], depth + 1),
            'right': self._build_tree(X[~left_idx], y[~left_idx], depth + 1)
        }

    def fit(self, X, y, n_features=None):
        self.feature_indices = np.random.choice(X.shape[1],
                                                n_features or int(np.sqrt(X.shape[1])), replace=False)
        self.tree = self._build_tree(X, y)

    def _predict_single(self, x, node):
        if 'class' in node:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        return self._predict_single(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.feature_importances = None

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.feature_importances = np.zeros(X.shape[1])
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            self._update_feature_importance(tree.tree)

        self.feature_importances /= np.sum(self.feature_importances)

    def _update_feature_importance(self, node):
        if 'feature' in node:
            self.feature_importances[node['feature']] += 1 - node['gini']
            self._update_feature_importance(node['left'])
            self._update_feature_importance(node['right'])

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),
                                   axis=0, arr=predictions)


# 数据加载与预处理
#iris = load_iris()
#X, y = iris.data, iris.target
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(r'D:\下载\iris\iris.data', header=None, names=columns)

# 数据清洗验证
print(f"原始数据量: {len(df)}条")
print("缺失值统计:", df.isnull().sum())
df.drop_duplicates(inplace=True)
print(f"去重后数据量: {len(df)}条")


le = LabelEncoder()
X = df.iloc[:, :-1].values
y = le.fit_transform(df.iloc[:, -1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
rf = RandomForest(n_estimators=100, max_depth=5)
rf.fit(X_train, y_train)

# 模型评估n
y_pred = rf.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.2%}")

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()

# 特征重要性可视化
features = columns[:-1]
importances = rf.feature_importances
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.array(features)[indices],hue=np.array(features)[indices], palette='viridis', legend=False)
plt.title('特征重要性评估')
plt.xlabel('重要性得分')
plt.ylabel('特征名称')
plt.show()
