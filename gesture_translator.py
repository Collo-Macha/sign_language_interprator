import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load the trained model
model = joblib.load("gesture_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Gesture Translator Running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    predicted_word = "No hand detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract features exactly like in training
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).reshape(1, -1)

            # Predict
            prediction = model.predict(landmarks)[0]
            predicted_word = prediction.upper()

            # Optional: Add confidence (distance to nearest neighbor)
            distances, _ = model.kneighbors(landmarks)
            confidence = 1 / (1 + distances[0][0])  # Simple confidence score

            if confidence < 0.5:  # Low confidence threshold
                predicted_word = "Uncertain"

    # Display the translated word
    cv2.putText(frame, f"Word: {predicted_word}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Gesture to Text Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()