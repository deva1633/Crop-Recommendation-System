# 🌾 Crop Recommendation System (Accuracy: 99.55%)

This project builds a **Crop Recommendation System** using **machine learning**, trained on the **Crop Recommendation Dataset** from Kaggle.  
It predicts the most suitable crop to grow based on **soil composition (N, P, K)** and **environmental conditions (temperature, humidity, pH, rainfall)**.

---

## 🚀 Project Overview

The model takes in agricultural and weather parameters to recommend the optimal crop for cultivation.  
By using feature scaling, encoding, and ensemble methods (XGBoost), this model achieves **99.55% prediction accuracy**.

---

## 📊 Dataset

**Source:** [Crop Recommendation Dataset (Kaggle)](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)  

| Feature | Description |
|----------|-------------|
| `N` | Nitrogen content in soil |
| `P` | Phosphorus content in soil |
| `K` | Potassium content in soil |
| `temperature` | Temperature in °C |
| `humidity` | Relative humidity (%) |
| `ph` | Soil pH value |
| `rainfall` | Annual rainfall (mm) |
| `label` | Recommended crop name |

---

## 🧠 ML Workflow

### 1️⃣ Import Required Libraries
```python
# ==========================================================
# 🌾 Crop Recommendation System | Accuracy: ~99.55%
# ==========================================================

# 1️⃣ Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------
# 2️⃣ Load Dataset
# ----------------------------------------------------------
df = pd.read_csv('/kaggle/input/crop-recommendation-dataset/Crop_recommendation.csv')
print(f"✅ Dataset Loaded Successfully — {df.shape[0]} rows, {df.shape[1]} columns")

# ----------------------------------------------------------
# 3️⃣ Explore Data
# ----------------------------------------------------------
print(df.info())
print(df.describe())

# Check for missing values
print("Missing Values in Each Column:\n", df.isnull().sum())

# Display unique crop labels
print("\nUnique Crops in Dataset:", df['label'].unique())

# ----------------------------------------------------------
# 4️⃣ Data Visualization (Exploratory Data Analysis)
# ----------------------------------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(y='label', data=df, order=df['label'].value_counts().index)
plt.title("Crop Frequency Distribution")
plt.show()

# ----------------------------------------------------------
# 5️⃣ Feature Selection & Encoding
# ----------------------------------------------------------
X = df.drop('label', axis=1)
y = df['label']

# Encode crop labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalize feature values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------
# 6️⃣ Train-Test Split
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print("✅ Data Split — Training:", X_train.shape, "Testing:", X_test.shape)

# ----------------------------------------------------------
# 7️⃣ Model Training (XGBoost Classifier)
# ----------------------------------------------------------
model = XGBClassifier(
    n_estimators=250,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(X_train, y_train)
print("✅ Model Training Completed")

# ----------------------------------------------------------
# 8️⃣ Model Evaluation
# ----------------------------------------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ----------------------------------------------------------
# 9️⃣ Cross-Validation (Optional)
# ----------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
print(f"📊 Cross-validation accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ----------------------------------------------------------
# 🔟 Predict New Data (Example)
# ----------------------------------------------------------
# Example input: [N, P, K, temperature, humidity, ph, rainfall]
sample_input = np.array([[90, 42, 43, 21.5, 80, 6.2, 202]])
sample_scaled = scaler.transform(sample_input)
predicted_crop = le.inverse_transform(model.predict(sample_scaled))[0]
print(f"🌱 Recommended Crop: {predicted_crop}")

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
