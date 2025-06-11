from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
import os

app = Flask(__name__)

# تحميل النموذج المدرب (يفترض أن الملف في نفس مجلد app.py)
MODEL_PATH = 'hand_gesture_model.h5'
model = load_model(MODEL_PATH)
gestures = ["Hello", "Good", "Bad", "thanks"]

# إعداد Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def extract_hand_landmarks(image_path):
    """استخراج معالم اليد من الصورة"""
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
    """التنبؤ بالإيماءة"""
    landmarks, error = extract_hand_landmarks(image_path)
    if error:
        return {"error": error}

    landmarks = landmarks.reshape(1, 21, 3)
    prediction = model.predict(landmarks)
    predicted_index = np.argmax(prediction)

    return {"gesture": gestures[predicted_index], "confidence": float(prediction[0][predicted_index]) * 100}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """رفع الصورة والتنبؤ"""
    if 'file' not in request.files:
        return jsonify({"error": "لم يتم رفع أي ملف."})

    file = request.files['file']
    filepath = os.path.join('static', 'uploaded_image.jpg')
    file.save(filepath)

    result = predict_gesture(filepath)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

