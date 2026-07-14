import pandas as pd
df=pd.read_csv("Teen_Mental_Health_Dataset.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df["gender"].unique())
print(df["platform_usage"].unique())
print(df["social_interaction_level"].unique())

print("\nCategorical Columns:")
print(df.select_dtypes(include=["object", "string"]).columns.tolist())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

encoders = {}
for column in ["gender", "platform_usage", "social_interaction_level"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le
X=df.drop('depression_label', axis=1)
y=df['depression_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

y_pred=model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,y_pred))
print("Classification Report:\n", classification_report(y_test,y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred))

import pickle

pickle.dump(model, open("mental_health_model.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))
print(df["gender"].unique())

print("Model Saved Successfully")
print("Encoders Saved Successfully")
