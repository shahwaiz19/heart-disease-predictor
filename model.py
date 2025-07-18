import pandas as pd
import joblib

import matplotlib.pyplot as plt


from  sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score , classification_report , confusion_matrix , precision_score , recall_score , f1_score

# Load the dataset

data = pd.read_csv("heart.csv")

# Fix text columns by converting to numbers
data = pd.get_dummies(data)

print(data.dtypes)


#print(data.head())

# data preprocessing check for missing values 

print(data.isnull().sum())

# define target
x = data.drop("HeartDisease",axis=1)
y = data["HeartDisease"] 

# split the data into training and testing sets
x_train,x_test,y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Initialize the model

model = RandomForestClassifier()
model.fit(x_train , y_train)

# make predictions
y_predict = model.predict(x_test)

# print the accuracy of the model
print("accuray" , accuracy_score(y_test , y_predict ))


print("classification report",classification_report(y_test,y_predict))

# Step 8: Plot feature importance
plt.figure(figsize=(10, 6))
feat_importance = pd.Series(model.feature_importances_, index=x.columns)
feat_importance.nlargest(10).plot(kind='barh')
plt.title("Important Features for Heart Disease Prediction")
plt.show()

joblib.dump(model, "heart_model.pkl")
joblib.dump(x.columns.tolist(), "model_features.pkl")
print("Model and features saved successfully.")