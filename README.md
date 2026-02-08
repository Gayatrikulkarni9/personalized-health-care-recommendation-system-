# personalized-health-care-recommendation-system-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

try:
    df = pd.read_csv("blood.csv")
except FileNotFoundError:
    print("File not found. Please upload the 'blood.csv.")
    uploaded = files.upload()
    if uploaded:
        # Assuming the user uploads the correct file
        for fn in uploaded.keys():
            print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')
            df = pd.read_csv(fn)
    else:
        print("No file was uploaded. Please upload the file to proceed.")
        df = None

if df is not None:
    print("Dataset loaded successfully.")


df.head()


df.info()
df.describe()


plt.figure(figsize=(6,4))
sns.histplot(df['Recency'], bins=20)
plt.title("Recency Distribution of Patients")
plt.show()


plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

sns.countplot(x="Class", data=df)
plt.title("Class Distribution")
plt.show()

X = df.drop("Class", axis=1)
y = df["Class"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_features = X.columns
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features)
    ])

from sklearn.ensemble import RandomForestClassifier

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.title("Confusion Matrix")
plt.show()

def generate_recommendation(patient_data):
    prediction = model_pipeline.predict(patient_data)

   
    
  mapping = {
        0: "No donation needed yet",
        1: "Ready for donation"
    }
    return mapping[prediction[0]]

example = pd.DataFrame({
    "Recency": [2],
    "Frequency": [5],
    "Monetary": [1250],
    "Time": [30]
})

# Adjust the mapping to reflect the actual classes (0 and 1) of the model's output
def generate_recommendation(patient_data):
    prediction = model_pipeline.predict(patient_data)

  mapping = {
        0: "No donation needed yet",
        1: "Ready for donation"
    }
    return mapping[prediction[0]]

print(generate_recommendation(example))


def generate_recommendation(patient_data):
    prediction = model_pipeline.predict(patient_data)

   mapping = {
        0: "No donation needed yet",
        1: "Ready for donation"
    }
    return mapping[prediction[0]]

example = pd.DataFrame({
    "Recency": [2],
    "Frequency": [5],
    "Monetary": [1250],
    "Time": [30]
})

print(generate_recommendation(example))


example = pd.DataFrame({
    "Recency": [2],
    "Frequency": [5],
    "Monetary": [1250],
    "Time": [30]
})

print(generate_recommendation(example))
























    
