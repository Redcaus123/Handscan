import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os

# Modell herunterladen falls nicht vorhanden
if not os.path.exists("hand_landmarker.task"):
    print("Lade Modell herunter...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "hand_landmarker.task"
    )
    print("Fertig!")

def get_fingers(landmarks):
    tips = [8, 12, 16, 20]
    base = [6, 10, 14, 18]
    fingers = []
    fingers.append(landmarks[4].x < landmarks[3].x)
    for tip, b in zip(tips, base):
        fingers.append(landmarks[tip].y < landmarks[b].y)
    return fingers

def erkenneGeste(fingers):
    daumen, zeige, mittel, ring, kleiner = fingers
    if all(fingers):                                    return "Alle offen"
    if not any(fingers):                                return "Faust"
    if zeige and not mittel:                            return "Zeigefinger"
    if zeige and mittel and not ring:                   return "Peace"
    if daumen and kleiner and not zeige:                return "Hang Loose"
    if daumen and not any([zeige,mittel,ring,kleiner]): return "Daumen hoch"
    return "Unbekannt"

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
geste = "Keine Hand"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        fingers = get_fingers(landmarks)
        geste = erkenneGeste(fingers)

        for lm in landmarks:
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        if geste == "Peace":
            print("Peace erkannt!")
        elif geste == "Faust":
            print("Faust erkannt!")
    else:
        geste = "Keine Hand"

    cv2.putText(frame, geste, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
    cv2.imshow("Hand Gesten", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
