
# Customer Churn Prediction
# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
# 2. Load Dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())
print(df.info())
print(df.describe())
print("Shape:", df.shape)

# 3. Target Variable Analysis
print(df["Churn"].value_counts())

# Convert target to numeric
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop ID column
df.drop("customerID", axis=1, inplace=True)
# 4. Encode Categorical Variables
df = pd.get_dummies(df, drop_first=True)
# 5. Split Features & Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 6. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 7. Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Predictions

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ================================
# 9. Evaluation Metrics
# ================================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ================================
# 10. Confusion Matrix Visualization
# ================================
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, cmap="Blues", values_format="d"
)
plt.title("Customer Churn Confusion Matrix")
plt.show()

# ================================
# 11. Predict for an Unseen Customer
# ================================
new_customer = X.iloc[[0]]  # unseen example from dataset
new_customer_scaled = scaler.transform(new_customer)

prediction = model.predict(new_customer_scaled)
probability = model.predict_proba(new_customer_scaled)

print("\nNew Customer Prediction:")
print("Churn Prediction:", "Likely to churn" if prediction[0] == 1 else "Likely to stay")
print("Churn Probability:", probability[0][1])
