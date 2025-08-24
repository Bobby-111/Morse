# morse_api.py (FastAPI version)
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI(title="Hand Gesture Recognition API", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your TFLite model
MODEL_PATH = "hand_gesture_model.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Labels mapping
labels_dict = {0: 'Dot', 1: 'Dash', 2: 'BlankSpace', 3: 'BackSpace', 4: 'Next'}

# Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


@app.get("/")
async def index():
    return {"message": "Welcome to the Hand Gesture Recognition API!"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # Convert to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return JSONResponse(content={
                "prediction": None,
                "confidence": 0.0,
                "message": "No hands detected"
            })

        hand_landmarks = results.multi_hand_landmarks[0]

        # Preprocess 21 landmarks → 42 features
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        data_aux = []
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_coords))
            data_aux.append(lm.y - min(y_coords))

        if len(data_aux) != 42:
            return JSONResponse(content={
                "prediction": None,
                "confidence": 0.0,
                "message": "Unexpected landmark length"
            })

        # Run inference
        input_data = np.array([data_aux], dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])

        pred_idx = int(np.argmax(output_data[0]))
        confidence = float(np.max(output_data[0]))
        prediction = labels_dict.get(pred_idx, str(pred_idx))

        return {
            "prediction": prediction,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
