# ============================================================
# Titanic Survival Prediction — Final Project
# Tools: Pandas, Seaborn, Matplotlib, Scikit-learn
# Models: Logistic Regression vs Decision Tree
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# STEP 1 — Load Data
# ============================================================

df = sns.load_dataset('titanic')
print("Dataset Shape:", df.shape)

# ============================================================
# STEP 2 — Clean Data
# ============================================================

# Fill missing values
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Drop unnecessary columns
df.drop(columns=['deck', 'embark_town', 'alive', 'who', 'adult_male', 'alone', 'class'], inplace=True)

# Convert text to numbers
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print("Missing values after cleaning:")
print(df.isnull().sum())

# ============================================================
# STEP 3 — EDA (Exploratory Data Analysis)
# ============================================================

plt.figure(figsize=(12, 4))

# Survival by Gender
plt.subplot(1, 2, 1)
df.groupby('sex')['survived'].mean().plot(kind='bar', color=['steelblue', 'salmon'])
plt.title('Survival Rate by Gender')
plt.xticks([0, 1], ['Male', 'Female'], rotation=0)
plt.ylabel('Survival Rate')

# Survival by Class
plt.subplot(1, 2, 2)
df.groupby('pclass')['survived'].mean().plot(kind='bar', color=['gold', 'silver', 'brown'])
plt.title('Survival Rate by Class')
plt.xticks([0, 1, 2], ['1st', '2nd', '3rd'], rotation=0)
plt.ylabel('Survival Rate')

plt.tight_layout()
plt.show()

# ============================================================
# STEP 4 — Prepare Data for Modeling
# ============================================================

X = df.drop(columns=['survived'])  # Features
y = df['survived']                 # Target

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# STEP 5 — Model 1: Logistic Regression
# ============================================================

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

# ============================================================
# STEP 6 — Model 2: Decision Tree
# ============================================================

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("\n=== Decision Tree ===")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

# ============================================================
# STEP 7 — Model Comparison Chart
# ============================================================

models = ['Logistic Regression', 'Decision Tree']
accuracies = [accuracy_score(y_test, lr_pred), accuracy_score(y_test, dt_pred)]

plt.bar(models, accuracies, color=['steelblue', 'salmon'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.75, 0.85)
plt.show()