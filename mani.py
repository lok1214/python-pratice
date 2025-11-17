"""
Hybrid Drowsy Driver Detection with Alarm
- Uses MediaPipe FaceMesh for eyes & mouth
- Uses a CNN for facial expression (sleepy, neutral, happy, etc.)
- Plays alarm when eyes stay closed too long
"""
#comments added

import cv2
import mediapipe as mp
import numpy as np
import threading
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# For alarm sound
from playsound import playsound

# ------------------- Alarm setup -------------------
def play_alarm():
    """Plays alarm sound in a separate thread"""
    if not os.path.exists("C:/Users/shymk/Downloads/alarm.mp3"):
        print("[WARNING] alarm.mp3 not found! Please add an alarm sound file in the same folder.")
        return
    playsound("C:/Users/shymk/Downloads/alarm.mp3")

def start_alarm():
    """Ensures alarm doesn’t overlap by running only once at a time"""
    global alarm_on
    if not alarm_on:
        alarm_on = True
        threading.Thread(target=play_alarm, daemon=True).start()

# ------------------- Load expression model -------------------
try:
    model = load_model("fer_model.h5")
    labels = ['angry','disgust','fear','happy','neutral','sad','surprise','sleepy']
except:
    model = None
    print("[INFO] Facial expression model not found (fer_model.h5). Expression detection disabled.")

# ------------------- Mediapipe setup -------------------
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
INNER_MOUTH = [78, 81, 13, 311, 308, 402]

EAR_THRESH = 0.25
MAR_THRESH = 0.6
EAR_CONSEC_FRAMES = 20
MAR_CONSEC_FRAMES = 15

eye_closed_frames = 0
yawn_frames = 0
alarm_on = False

# ------------------- Utility functions -------------------
def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye_points):
    A = euclidean_dist(eye_points[1], eye_points[5])
    B = euclidean_dist(eye_points[2], eye_points[4])
    C = euclidean_dist(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth_points):
    A = euclidean_dist(mouth_points[1], mouth_points[2])
    B = euclidean_dist(mouth_points[4], mouth_points[5])
    C = euclidean_dist(mouth_points[0], mouth_points[3])
    return (A + B) / (2.0 * C)

# ------------------- Initialize camera -------------------
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        h, w, _ = frame.shape
        face_crop = None

        if result.multi_face_landmarks:
            for landmarks in result.multi_face_landmarks:
                points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]

                left_eye = [points[i] for i in LEFT_EYE]
                right_eye = [points[i] for i in RIGHT_EYE]
                mouth = [points[i] for i in INNER_MOUTH]

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                mar = mouth_aspect_ratio(mouth)

                # Draw landmarks
                for pt in left_eye + right_eye + mouth:
                    cv2.circle(frame, pt, 2, (0, 255, 255), -1)

                # Detect drowsiness by eye & mouth
                if ear < EAR_THRESH:
                    eye_closed_frames += 1
                else:
                    eye_closed_frames = 0
                    alarm_on = False  # reset alarm

                if mar > MAR_THRESH:
                    yawn_frames += 1
                else:
                    yawn_frames = 0

                # Crop face region for expression detection
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x1, x2 = max(0, min(x_coords)), min(w, max(x_coords))
                y1, y2 = max(0, min(y_coords)), min(h, max(y_coords))
                face_crop = frame[y1:y2, x1:x2]

                # Alerts
                if eye_closed_frames >= EAR_CONSEC_FRAMES:
                    cv2.putText(frame, "EYES CLOSED!", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    start_alarm()

                if yawn_frames >= MAR_CONSEC_FRAMES:
                    cv2.putText(frame, "YAWNING!", (30, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # ------------------- Expression detection -------------------
        if model is not None and face_crop is not None and face_crop.size > 0:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48, 48))
            roi = gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi, verbose=0)[0]
            label = labels[np.argmax(preds)]
            conf = preds[np.argmax(preds)]

            cv2.putText(frame, f"Expr: {label} ({conf*100:.1f}%)", (30, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # If face looks sleepy/drowsy
            if label.lower() in ["sleepy", "sad", "neutral"] and conf > 0.6:
                cv2.putText(frame, "DROWSY FACE DETECTED!", (30, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # ------------------------------------------------------------
        cv2.imshow("Drowsy Driver Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

