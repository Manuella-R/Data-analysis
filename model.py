# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load or create the dataset
data = pd.read_csv('ai_job_market_insights.csv')
data = data.dropna() 
df = pd.DataFrame(data)

# Encode categorical variables
encoder = LabelEncoder()
df["Automation_Risk_Encoded"] = encoder.fit_transform(df["Automation_Risk"])
df["Company_Size_Encoded"] = encoder.fit_transform(df["Company_Size"])
df["AI_Adoption_Level_Encoded"] = encoder.fit_transform(df["AI_Adoption_Level"])

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[["Automation_Risk_Encoded", "Company_Size_Encoded", "AI_Adoption_Level_Encoded"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Features")
plt.show()

# Prepare features and target variable
X = df[["Automation_Risk_Encoded", "Company_Size_Encoded"]]
y = df["AI_Adoption_Level_Encoded"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Dynamically handle missing classes in the test set
unique_classes = sorted(set(y_test))  # Get unique classes in y_test
target_names = [encoder.inverse_transform([cls])[0] for cls in unique_classes]

# Evaluate the model
classification_rep = classification_report(y_test, y_pred, target_names=target_names)
conf_matrix = confusion_matrix(y_test, y_pred)

# Feature Importance
feature_importance = model.feature_importances_

# Visualization of Feature Importance
plt.figure(figsize=(8, 5))
sns.barplot(x=X.columns, y=feature_importance, palette="viridis")
plt.title("Feature Importance in Predicting AI Adoption Levels")
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.show()

# Display Outputs
print("Classification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)
