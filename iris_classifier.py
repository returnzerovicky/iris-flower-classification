import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)

# ==========================
# Load Dataset
# ==========================

iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add Species Column
df["Species"] = iris.target

# Convert numeric labels to names
df["Species"] = df["Species"].replace({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

# ==========================
# Display Dataset
# ==========================

print("=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumn Names")
print(df.columns)

print("\nInformation")
df.info()

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# ==========================
# Count Plot
# ==========================

plt.figure(figsize=(6,4))
sns.countplot(x="Species", data=df)
plt.title("Number of Flowers in Each Species")
plt.show()

# ==========================
# Pair Plot
# ==========================

sns.pairplot(df, hue="Species")
plt.show()

# ==========================
# Correlation Heatmap
# ==========================

plt.figure(figsize=(8,6))
sns.heatmap(
    df.drop("Species", axis=1).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation")
plt.show()

# ==========================
# Machine Learning
# ==========================

# Reload numeric labels
X = iris.data
y = iris.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Labels Shape:", y_train.shape)
print("Testing Labels Shape:", y_test.shape)

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFirst Training Sample After Scaling:")
print(X_train[0])

# ==========================
# KNN Model
# ==========================

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ==========================
# Predictions
# ==========================

y_pred = model.predict(X_test)

print("\nPredicted Labels:")
print(y_pred)

print("\nActual Labels:")
print(y_test)

# ==========================
# Evaluation
# ==========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score")
print(accuracy)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

f1 = f1_score(y_test, y_pred, average="weighted")

print("Weighted F1 Score:", f1)

# ==========================
# Confusion Matrix Heatmap
# ==========================

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==========================
# Test on New Flower
# ==========================

sample = [[5.1, 3.5, 1.4, 0.2]]

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

print("\nPrediction for Sample Flower:")

print("Predicted Species:", iris.target_names[prediction[0]])