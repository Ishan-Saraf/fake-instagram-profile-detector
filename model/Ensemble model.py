import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the dataset
data = pd.read_csv("../train.csv")

# Assuming your dataset has features (X) and labels (y)
X = data.drop('fake', axis=1)
y = data['fake']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load the pre-trained SVM model using joblib
svm_model = joblib.load(r'C:\Users\JD\PycharmProjects\fake instagram account detector\model\model.pkl')

# Create other base models
rf_model = RandomForestClassifier(n_estimators=100)
ann_model = load_model(r'C:\Users\JD\PycharmProjects\fake instagram account detector\model\modelA.h5')

# Create a stacking ensemble
base_models = [('svm', svm_model), ('rf', rf_model), ('ann', ann_model)]
meta_model = LogisticRegression()

stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# Fit the stacking model on the training data
stacking_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
stacking_predictions = stacking_model.predict(X_test_scaled)

# Evaluate the stacking model
accuracy = accuracy_score(y_test, stacking_predictions)
print("Stacking Model Accuracy:", accuracy)
