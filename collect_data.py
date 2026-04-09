import cv2
import mediapipe as mp
import numpy as np
import os

# MediaPipe setup (works with the version you installed)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

DATA_DIR = "gesture_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ←←← Change or add your gestures here
gestures = ["hello", "thank_you", "yes", "no", "please", "ok"]

cap = cv2.VideoCapture(0)

print("=== Gesture Data Collection ===")
print("Show one gesture clearly in front of the camera.")
print("Press the number key to save a sample:")
for i, g in enumerate(gestures):
    print(f"   {i} → {g}")
print("\nPress 'q' to quit when done.\n")

sample_count = {g: 0 for g in gestures}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 63 landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks)

    # Show sample counts on screen
    y_offset = 30
    for i, g in enumerate(gestures):
        cv2.putText(frame, f"{i}: {g} ({sample_count[g]} samples)", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30

    cv2.imshow("Data Collection (Press number key)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord(str(i)) for i in range(len(gestures))]:
        idx = int(chr(key))
        gesture_name = gestures[idx]
        file_path = os.path.join(DATA_DIR, f"{gesture_name}.npy")

        if 'landmarks' in locals():
            if os.path.exists(file_path):
                existing = np.load(file_path)
                existing = np.vstack((existing, landmarks))
                np.save(file_path, existing)
            else:
                np.save(file_path, landmarks.reshape(1, -1))
            
            sample_count[gesture_name] += 1
            print(f"✓ Saved sample #{sample_count[gesture_name]} for '{gesture_name}'")
        else:
            print("No hand detected! Show your hand clearly.")

cap.release()
cv2.destroyAllWindows()
print("\nData collection finished. Check the 'gesture_data' folder.")