import numpy as np
import pandas as pd
import os
from scipy.stats import kurtosis, skew
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 🔹 修改資料集路徑，確保在 Docker 環境內可用
DATASET_PATH = "/data1/zoey/fall_detection_project/UCI_HAR_Dataset"

# 讀取 `features.txt`（特徵名稱）
features_path = os.path.join(DATASET_PATH, "features.txt")
feature_names = pd.read_csv(features_path, sep="\s+", header=None, usecols=[1], names=["feature"]).squeeze("columns").tolist()

# 讀取 `activity_labels.txt`（標籤名稱）
labels_path = os.path.join(DATASET_PATH, "activity_labels.txt")
activity_labels = pd.read_csv(labels_path, sep="\s+", header=None, names=["id", "activity"])

# 讀取 `train/` 和 `test/` 數據
def load_data(folder):
    x_path = os.path.join(DATASET_PATH, folder, f"X_{folder}.txt")
    y_path = os.path.join(DATASET_PATH, folder, f"y_{folder}.txt")
    subject_path = os.path.join(DATASET_PATH, folder, f"subject_{folder}.txt")

    X = pd.read_csv(x_path, sep="\s+", header=None, names=feature_names)
    y = pd.read_csv(y_path, sep="\s+", header=None, names=["activity"])
    subjects = pd.read_csv(subject_path, sep="\s+", header=None, names=["subject"])

    return X, y["activity"], subjects["subject"]

# 讀取訓練 & 測試數據
X_train, y_train, subjects_train = load_data("train")
X_test, y_test, subjects_test = load_data("test")

# 訓練 SVM
svm_model = SVC(kernel="linear")
svm_model.fit(X_train, y_train)

# 預測測試集
y_pred = svm_model.predict(X_test)

# 計算評估指標
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# 顯示結果
print(f"✅ 準確率 (Accuracy): {accuracy:.4f}")
print(f"✅ 精確率 (Precision): {precision:.4f}")
print(f"✅ 召回率 (Recall): {recall:.4f}")
print(f"✅ F1 分數: {f1:.4f}")

# 繪製混淆矩陣
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
