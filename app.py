# morse_api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)

# Load your TFLite model
MODEL_PATH = "morse_model.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Labels mapping (from your Python file)
labels_dict = {0: 'Dot', 1: 'Dash', 2: 'BlankSpace', 3: 'BackSpace', 4: 'Next'}

# Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # Convert to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return jsonify({"prediction": None, "confidence": 0.0, "message": "No hands detected"})

    hand_landmarks = results.multi_hand_landmarks[0]

    # Preprocess 21 landmarks → 42 features
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    data_aux = []
    for lm in hand_landmarks.landmark:
        data_aux.append(lm.x - min(x_coords))
        data_aux.append(lm.y - min(y_coords))

    if len(data_aux) != 42:
        return jsonify({"prediction": None, "confidence": 0.0, "message": "Unexpected landmark length"})

    # Run inference
    input_data = np.array([data_aux], dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    pred_idx = int(np.argmax(output_data[0]))
    confidence = float(np.max(output_data[0]))
    prediction = labels_dict.get(pred_idx, str(pred_idx))

    return jsonify({
        "prediction": prediction,
        "confidence": confidence
    })


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=False)
