import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("/content/Students.csv")


print("Data Preview:")
print(data.head())


data['Financial_Aid_Status'] = data['Financial_Aid_Status'].map({'Yes': 1, 'No': 0})
data['Engagement_Level'] = data['Engagement_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
data['Age_Group'] = data['Age_Group'].map({'18-22': 0, '23-27': 1, '28-32': 2})


X = data.drop(['Student_Code'], axis=1)
y = np.random.choice([0, 1], size=(len(data),))  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))


feature_importance = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance in Predicting Student Enrollment")
plt.show()
