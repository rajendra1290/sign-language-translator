from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
import asyncio
from collections import deque
import time

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load ASL model
model = load_model("model/asl_cnn_model.h5")
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']  # A-Z

# Model input shape
MODEL_H, MODEL_W, MODEL_C = 50, 50, 1  # grayscale 50x50

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Stability filter
    STABILITY_COUNT = 2
    recent_preds = deque(maxlen=STABILITY_COUNT)

    last_prediction_time = 0
    prediction_interval = 3  # seconds

    while True:
        try:
            data = await websocket.receive_text()
        except Exception:
            break

        # Drain any queued frames so we always process the most recent
        try:
            while True:
                more = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                data = more
        except asyncio.TimeoutError:
            pass
        except Exception:
            break

        # Decode frame
        img_data = base64.b64decode(data.split(",")[1])
        npimg = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        letter = "-"

        # -----------------------------
        # HAND DETECTION (Contour-based)
        # -----------------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Tighter HSV range for skin tone (more selective)
        lower = np.array([0, 20, 70], dtype="uint8")
        upper = np.array([30, 150, 255], dtype="uint8")
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Apply morphological operations to clean noise and fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size and shape to detect only hand-like objects
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Reject very small contours (noise)
            if area < 500:
                continue
            # Reject very large contours (unlikely to be a hand in frame)
            if area > frame.shape[0] * frame.shape[1] * 0.6:
                continue
            # Check aspect ratio (hand is roughly rectangular/square)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            # Hand aspect ratio typically between 0.4 and 2.5
            if 0.4 <= aspect_ratio <= 2.5:
                valid_contours.append(contour)

        contours = valid_contours

        current_time = time.time()
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            # Draw green rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            hand_roi = frame[y:y+h, x:x+w]

            if hand_roi.size > 0 and (current_time - last_prediction_time >= prediction_interval):
                # Preprocess: grayscale, resize, normalize
                gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (MODEL_W, MODEL_H))
                normalized = resized / 255.0
                reshaped = normalized.reshape(1, MODEL_H, MODEL_W, MODEL_C)

                try:
                    pred = model.predict(reshaped)
                    raw_letter = labels[np.argmax(pred)]

                    # Stability check
                    recent_preds.append(raw_letter)
                    if len(recent_preds) == recent_preds.maxlen and all(p == raw_letter for p in recent_preds):
                        letter = raw_letter
                        last_prediction_time = current_time
                    else:
                        letter = "-"
                except Exception as e:
                    print("Prediction error:", e)
                    letter = "-"

        # Encode processed frame and send back
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')

        await websocket.send_json({
            "letter": letter,
            "frame": encoded_frame
        })

        # Drain any queued frames quickly
        try:
            while True:
                more = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                continue
        except asyncio.TimeoutError:
            pass
        except Exception:
            break
