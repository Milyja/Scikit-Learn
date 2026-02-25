import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Safety check
if len(data) == 0:
    print("Dataset kosong! Jalankan ekstraksi data terlebih dahulu.")
    exit()

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.1, shuffle=True
)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f"{score * 100:.2f}% of samples were classified correctly!")

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
