# 📊 Customer Churn Prediction System

An end-to-end Machine Learning project that predicts whether a customer is likely to churn using classification models, data analysis, and visualization.

---

## 🚀 Overview

Customer churn prediction is crucial for businesses to retain customers and reduce revenue loss.
This project uses **Machine Learning models** to identify high-risk customers based on behavioral and demographic data.

---

## 🔥 Features

* 📂 Data preprocessing & cleaning
* 🔢 Feature engineering using one-hot encoding
* 🤖 Model training using:

  * Random Forest
  * Logistic Regression
* ⚖️ Handling class imbalance using `class_weight`
* 📊 Model evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
* 🎯 Threshold tuning to improve churn detection
* 📈 Data visualization:

  * Feature importance
  * Confusion matrix
  * Churn distribution
  * Tenure vs churn
  * Monthly charges vs churn

---

## 🧠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

## 📁 Project Structure

```bash
churn-project/
│── churn.csv
│── churn_analysis.py
│── README.md
```

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## ▶️ Run the Project

```bash
python churn_analysis.py
```

---

## 📊 Model Performance

### 🔹 Random Forest

* Accuracy: ~80%
* Strong overall performance
* Provides feature importance

### 🔹 Logistic Regression

* Accuracy: ~77–78%
* Improved **recall for churn detection**
* Better for business use

---

## 🎯 Key Improvements

* Applied **class weighting** to handle imbalanced data
* Used **threshold tuning (0.5 → 0.4)** to improve recall
* Reduced false negatives (missed churn cases)

---

## 📈 Key Insights

* Customers with **low tenure** are more likely to churn
* **High monthly charges** increase churn probability
* **Month-to-month contracts** show higher churn risk
* Lack of services like **online security and tech support** contributes to churn

---

## 📸 Visualizations

* Feature Importance Chart
* Confusion Matrix
* Churn Distribution
* Tenure vs Churn
* Monthly Charges vs Churn

<img width="1536" height="754" alt="Figure_1" src="https://github.com/user-attachments/assets/405e459f-319c-4489-9e69-a3da69fdab79" />

<img width="1536" height="754" alt="Figure_2" src="https://github.com/user-attachments/assets/3f221979-83e2-4d4f-94ea-fb70df40cafc" />

<img width="1536" height="754" alt="Figure_3" src="https://github.com/user-attachments/assets/83c78eb2-bd54-4d66-9b18-daca2cf59653" />

<img width="1536" height="754" alt="Figure_4" src="https://github.com/user-attachments/assets/caf80903-e15c-4a65-a7c7-a9bde1ffee35" />

<img width="1536" height="754" alt="Figure_5" src="https://github.com/user-attachments/assets/b2e8674c-b19d-4e8c-b26f-62f09bb7dad5" />

---

## 💡 Business Impact

* Helps companies identify high-risk customers
* Enables targeted retention strategies
* Reduces potential revenue loss

---

## 🚀 Future Improvements

* Hyperparameter tuning (GridSearchCV)
* Deployment using Flask / Streamlit
* Real-time prediction dashboard
* Model saving using joblib

---

## 👨‍💻 Author

**Maurya Solanki**
