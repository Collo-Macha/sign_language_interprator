import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

DATA_DIR = "gesture_data"
gestures = ["hello", "thank_you", "yes", "no", "please", "ok"]

X = []
y = []

for i, gesture in enumerate(gestures):
    file_path = os.path.join(DATA_DIR, f"{gesture}.npy")
    if os.path.exists(file_path):
        data = np.load(file_path)
        X.append(data)
        y.extend([gesture] * len(data))

X = np.vstack(X)
y = np.array(y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(model, "gesture_model.pkl")
print("Model saved as 'gesture_model.pkl'")