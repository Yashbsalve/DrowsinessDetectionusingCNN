import threading
import time
import sqlite3
from datetime import datetime
import torch
import cv2
import dlib
import imutils
from flask import Flask, render_template_string
from imutils import face_utils
from scipy.spatial import distance as dist
import simpleaudio as sa
import torchvision.transforms as transforms
from cnn_model import DrowsinessCNN
from PIL import Image  # ← Added import

# ─── CONFIG ────────────────────────────────────────────────────────────────
EAR_THRESHOLD   = 0.25
DROWSY_SECONDS  = 5
DB_PATH         = "drowsiness.db"
VIDEO_SOURCE    = 0
PREDICTOR_PATH  = "shape_predictor_68_face_landmarks.dat"
HOST, PORT      = "0.0.0.0", 5000
MODEL_PATH      = "cnn_drowsiness.pth"

# ─── SETUP DB ──────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            ts   TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_detection():
    ts = datetime.now().isoformat(sep=' ', timespec='seconds')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO detections (ts) VALUES (?)", (ts,))
    conn.commit()
    conn.close()

# ─── DROWSINESS THREAD ─────────────────────────────────────────────────────
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

cnn_model = DrowsinessCNN()
cnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
cnn_model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

def detection_loop():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    vs = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
    if not vs.isOpened():
        print(f"[ERROR] Could not open video source {VIDEO_SOURCE}")
        return
    time.sleep(1.0)

    eye_closed_start = None
    alert_triggered = False

    while True:
        ret, frame = vs.read()
        if not ret:
            print("[ERROR] Frame grab failed")
            break

        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

            # Extract and preprocess eye region for CNN
            y1 = min(shape[lStart][1], shape[lEnd][1])
            y2 = max(shape[lStart][1], shape[lEnd][1])
            x1 = min(shape[lStart][0], shape[rStart][0])
            x2 = max(shape[lStart][0], shape[rEnd][0])
            eye_crop = frame[y1:y2, x1:x2]

            if eye_crop.size == 0:
                continue  # skip empty crop

            # Convert BGR to RGB and to PIL Image
            eye_rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
            eye_pil = Image.fromarray(eye_rgb)
            eye_tensor = transform(eye_pil).unsqueeze(0)

            # Predict using CNN
            output = cnn_model(eye_tensor)
            _, predicted = torch.max(output, 1)

            if predicted.item() == 1:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                else:
                    elapsed = time.time() - eye_closed_start
                    if elapsed >= DROWSY_SECONDS and not alert_triggered:
                        print("Drowsiness Detected!")
                        try:
                            mp3 = sa.WaveObject.from_wave_file("buzzer.wav")
                            mp3.play()
                            log_detection()
                            alert_triggered = True
                        except Exception as e:
                            print("[ERROR] Sound error:", e)
            else:
                eye_closed_start = None
                alert_triggered = False

            # Visuals
            leftHull = cv2.convexHull(leftEye)
            rightHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if alert_triggered else (255, 255, 255), 2)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.release()
    cv2.destroyAllWindows()

# ─── WEB SERVER ───────────────────────────────────────────────────────────
app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Drowsiness Log</title>
  <style>
    body { font-family: sans-serif; }
    table { border-collapse: collapse; width: 60%; margin: 20px 0; }
    th, td { border: 1px solid #666; padding: 8px; text-align: left; }
  </style>
</head>
<body>
  <h2>Drowsiness Events</h2>
  <table>
    <thead><tr><th>#</th><th>Timestamp</th></tr></thead>
    <tbody>
      {% for row in rows %}
      <tr><td>{{ row[0] }}</td><td>{{ row[1] }}</td></tr>
      {% endfor %}
    </tbody>
  </table>
  <p>Last updated: {{ now }}</p>
  <script>
    setTimeout(() => location.reload(), 10000);
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, ts FROM detections ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return render_template_string(TEMPLATE, rows=rows, now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# ─── MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    app.run(host=HOST, port=PORT, threaded=True)
