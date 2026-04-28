# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# ==============================
# 2. LOAD DATA
# ==============================
df = pd.read_csv("churn.csv")

# ==============================
# 3. PREPROCESSING
# ==============================
df.drop("customerID", axis=1, inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df = pd.get_dummies(df)

# ==============================
# 4. SPLIT DATA
# ==============================
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 5. RANDOM FOREST (BASELINE)
# ==============================
rf_model = RandomForestClassifier(class_weight="balanced")
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n===== RANDOM FOREST =====")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# ==============================
# 6. FEATURE IMPORTANCE
# ==============================
importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# ==============================
# 7. VISUALIZE FEATURE IMPORTANCE
# ==============================
top_features = feature_importance.head(10)

plt.figure()
plt.barh(top_features["Feature"], top_features["Importance"])
plt.title("Top 10 Feature Importance")
plt.gca().invert_yaxis()
plt.show()

# ==============================
# 8. LOGISTIC REGRESSION (IMPROVED MODEL)
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_model.fit(X_train_scaled, y_train)

# Default prediction (0.5 threshold)
lr_pred = lr_model.predict(X_test_scaled)

print("\n===== LOGISTIC REGRESSION (DEFAULT) =====")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# ==============================
# 9. CONFUSION MATRIX (DEFAULT)
# ==============================
cm = confusion_matrix(y_test, lr_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix (Default Threshold)")
plt.show()

# ==============================
# 10. THRESHOLD TUNING (IMPORTANT)
# ==============================
probs = lr_model.predict_proba(X_test_scaled)[:, 1]

# Change threshold from 0.5 → 0.4
custom_pred = (probs > 0.4).astype(int)

print("\n===== LOGISTIC REGRESSION (THRESHOLD = 0.4) =====")
print(classification_report(y_test, custom_pred))

# Confusion Matrix (Improved)
cm2 = confusion_matrix(y_test, custom_pred)

disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp2.plot()
plt.title("Confusion Matrix (Threshold = 0.4)")
plt.show()

# ==============================
# 11. EDA VISUALIZATIONS
# ==============================

# Churn distribution
plt.figure()
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

# Tenure vs Churn
plt.figure()
sns.boxplot(x="Churn", y="tenure", data=df)
plt.title("Tenure vs Churn")
plt.show()

# Monthly Charges vs Churn
plt.figure()
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# ==============================
# 12. SAMPLE PREDICTION
# ==============================
sample = X_test.iloc[0:1]
sample_scaled = scaler.transform(sample)

prediction = lr_model.predict(sample_scaled)[0]
probability = lr_model.predict_proba(sample_scaled)[0][1]

print("\n===== SAMPLE PREDICTION =====")
print("Prediction:", "Churn" if prediction == 1 else "Stay")
print("Churn Probability:", round(probability * 100, 2), "%")