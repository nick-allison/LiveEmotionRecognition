import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO

# ─── CONFIG ───────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH = 'weights.pt'
LABELS = ['angry','disgust','fear','happy','neutral','sad','ahegao','surprise']

# must be a multiple of the model’s max stride (32), so 64 is the next up from 48
IMG_SIZE = 64  
n_ms     = 200  # grab a frame ~ every 200 ms (~5 FPS)

# big‐label parameters
FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 2.0
THICKNESS  = 4

# small‐distribution text parameters
D_FONT       = cv2.FONT_HERSHEY_SIMPLEX
D_FONT_SCALE = 0.6
D_THICKNESS  = 1
D_LINE_HEIGHT = 18  # pixels between lines

# device: MPS on mac, else CPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#Comment the above line and uncomment the below line for Windows or Linux:
#device = torch.device('cuda' if torch.backends.mps.is_available() else 'cpu')

# ─── LOAD YOLO MODEL ─────────────────────────────────────────────────────────
model = YOLO(YOLO_MODEL_PATH)  # classification head
model.to(device).eval()

# ─── SET UP WEBCAM ───────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

last_time = 0
while True:
    now = time.time() * 1000
    if now - last_time < n_ms:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    last_time = now

    ret, frame = cap.read()
    if not ret:
        break

    # ─── CREATE LOW-RES INPUT ──────────────────────────────────────────────────
    # resize your HD frame down to exactly IMG_SIZE × IMG_SIZE
    small_img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # ─── INFERENCE ─────────────────────────────────────────────────────────────
    # feed small_img (BGR) directly; YOLO will use that resolution
    results = model(small_img, imgsz=IMG_SIZE, device=device, verbose=False)[0]
    p       = results.probs                # Probs object
    pred    = int(p.top1)                  # top-1 index
    conf    = float(p.top1conf)            # top-1 confidence
    probs   = p.data.cpu().numpy()              # full softmax distribution

    # ─── OVERLAY: DISTRIBUTION ────────────────────────────────────────────────
    # draw a semi-transparent box behind the text block
    h, w = frame.shape[:2]
    dist_box_h = D_LINE_HEIGHT * (len(LABELS) + 1)
    cv2.rectangle(frame,
                  (5, 5),
                  (250, 5 + dist_box_h),
                  (0, 0, 0, 100),
                  cv2.FILLED)

    # write each class + % on the HD frame
    for i, label in enumerate(LABELS):
        text = f"{label}: {probs[i]*100:5.1f}%"
        y = 5 + (i+1) * D_LINE_HEIGHT
        cv2.putText(frame, text,
                    (10, y),
                    D_FONT, D_FONT_SCALE, (255,255,255),
                    D_THICKNESS, cv2.LINE_AA)

    # ─── OVERLAY: TOP LABEL ────────────────────────────────────────────────────
    emotion = LABELS[pred].upper()
    (tw, th), baseline = cv2.getTextSize(emotion, FONT, FONT_SCALE, THICKNESS)
    x, y = 10, h - 10
    # black filled box
    cv2.rectangle(frame,
                  (x-5, y+baseline+5),
                  (x+tw+5,  y-th-5),
                  (0, 0, 0),
                  cv2.FILLED)
    # white bold text
    cv2.putText(frame, emotion,
                (x, y),
                FONT, FONT_SCALE,
                (255,255,255), THICKNESS, cv2.LINE_AA)

    # ─── SHOW WINDOWS ─────────────────────────────────────────────────────────
    cv2.imshow('Live Emotion Recognition', frame)

    # upscale the small input so you can actually see it
    input_display = cv2.resize(small_img, (IMG_SIZE*4, IMG_SIZE*4), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Model Input (low-res)', input_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()