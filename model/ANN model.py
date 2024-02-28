from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score

# Importing and setting up the dataset:
data = pd.read_csv(r"C:\Users\JD\PycharmProjects\fake instagram account detector\train.csv")

# Preparing Data to Train the Model

X_train = data.drop(columns=['fake'])
X_test = data.drop(columns=['fake'])

y_train = data['fake']
y_test = data['fake']

# Scale the data before training the model

scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

y_train = tf.keras.utils.to_categorical(y_train, num_classes = 2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes = 2)

# Building and Training Deep Training Model

model = Sequential()
model.add(Dense(50, input_dim=11, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

epochs_hist = model.fit(X_train, y_train, epochs = 50,  verbose = 1, validation_split = 0.1)

# Access the Performance of the model

print(epochs_hist.history.keys())

predicted = model.predict(X_test)

predicted_value = []
test = []
for i in predicted:
    predicted_value.append(np.argmax(i))

for i in y_test:
    test.append(np.argmax(i))

print(classification_report(test, predicted_value))

# Save the entire model to a file using Keras
# model.save('model2A.h5')

# Load the model from the file
loaded_model = load_model('model2A.h5')

# Make predictions with the loaded model
predictions = loaded_model.predict(X_test)
predicted_labels = [1 if pred[1] > 0.5 else 0 for pred in predictions]
true_labels = [1 if label[1] > 0.5 else 0 for label in y_test]

# Displaying classification report and accuracy:
# print("Classification Report:\n", classification_report(true_labels, predicted_labels))
# print("Accuracy Score:", accuracy_score(true_labels, predicted_labels))
