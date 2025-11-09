# Crop-Recommendation-System
This project builds a **Crop Recommendation System** using **machine learning**, trained on the **Crop Recommendation Dataset** from Kaggle.   It predicts the most suitable crop to grow based on **soil composition (N, P, K)** and **environmental conditions (temperature, humidity, pH, rainfall)**.


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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
2️⃣ Data Preprocessing
Checked for missing values and data consistency
Encoded categorical labels (LabelEncoder)
Scaled numeric features using MinMaxScaler
3️⃣ Model Training
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
4️⃣ Evaluation
Achieved 99.55% accuracy on test data
Cross-validation confirms high model reliability
Confusion matrix and classification report show robust results across all crop classes
📈 Visualizations
Heatmap of feature correlations
Distribution plots of N, P, K, and pH
Crop frequency count bar chart
Confusion matrix of predicted vs actual crops
🧩 Model Performance
Metric	Score
Accuracy	99.55%
Precision	99%
Recall	99%
F1-Score	99%
🌱 Predicted Crops Example
N	P	K	temperature	humidity	ph	rainfall	Predicted Crop
90	42	43	21.5	80	6.2	202	rice
45	30	20	25.1	60	5.8	85	maize
120	55	65	22.3	75	6.5	190	sugarcane
⚙️ Technologies Used
Python 3.10+
Pandas, NumPy – Data manipulation
Matplotlib, Seaborn – Visualization
Scikit-learn – Model building and evaluation
XGBoost – Final ML classifier
🧪 Results Summary
The XGBoost Classifier outperformed other tested models like SVM, Decision Tree, and Random Forest.
Minimal overfitting with consistent validation accuracy.
Can be extended into a Flask web app or mobile advisory tool for farmers.
🧭 Future Enhancements
Integrate real-time weather data APIs
Deploy as a web app (Streamlit / Flask)
Add geo-location-based crop suggestions
Incorporate economic factors like crop price trends
🧑‍💻 Author & Credits
Originally developed by Kaggle contributors and modified for analysis.
Adapted and explored by Dev Senthilkumar for research and learning.
Dataset Credit: Kaggle – Crop Recommendation Dataset
