import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='./hand_gesture_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

labels_dict = {'0': 'Dot', '1': 'Dash', '2': 'BlankSpace', '3': 'BackSpace', '4': 'Next'}
display_map = {'Dot': '.', 'Dash': '-', 'BlankSpace': ' '}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {e}")

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    data_aux = []
    x_ = []
    y_ = []
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    hands.close()

    if not results.multi_hand_landmarks:
        return JSONResponse(content={"prediction": None, "confidence": 0.0, "message": "No hand detected"})

    for hand_landmarks in results.multi_hand_landmarks:
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

    if len(data_aux) != 42:
        return JSONResponse(content={"prediction": None, "confidence": 0.0, "message": "Hand not properly detected"})

    input_data = np.array([data_aux], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_idx = np.argmax(output_data[0])
    confidence = float(np.max(output_data[0]))

    if confidence > 0.85:
        predicted_class = label_encoder.classes_[predicted_class_idx]
        predicted_character = labels_dict[str(predicted_class)]
        return JSONResponse(content={"prediction": predicted_character, "confidence": confidence})
    else:
        return JSONResponse(content={"prediction": None, "confidence": confidence, "message": "Low confidence"})

@app.get("/")
def root():
    return {"message": "Hand Gesture FastAPI is running!"}

