from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os

app = Flask(__name__)

# تحميل نموذج TFLite
TFLITE_MODEL_PATH = 'hand_gesture_model.tflite'
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

gestures = ["Hello", "Good", "Bad", "thanks"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def extract_hand_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, "تعذر تحميل الصورة."

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None, "لم يتم اكتشاف اليد."

    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = [coord for point in hand_landmarks.landmark for coord in (point.x, point.y, point.z)]
    return np.array(landmarks, dtype=np.float32), None

def predict_gesture(image_path):
    landmarks, error = extract_hand_landmarks(image_path)
    if error:
        return {"error": error}

    input_data = landmarks.reshape(1, 21, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_index = np.argmax(prediction)
    return {
        "gesture": gestures[predicted_index],
        "confidence": float(prediction[predicted_index]) * 100
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "لم يتم رفع أي ملف."})
    file = request.files['file']
    filepath = os.path.join('static', 'uploaded_image.jpg')
    file.save(filepath)
    result = predict_gesture(filepath)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
