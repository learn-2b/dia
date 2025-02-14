from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("best_model.keras")

class_labels = {
    0: "No DR - العين سليمة",
    1: "Mild DR - اعتلال شبكية بسيط",
    2: "Moderate DR - اعتلال شبكية متوسط",
    3: "Severe DR - اعتلال شبكية شديد",
    4: "Proliferative DR - اعتلال شبكية خطير ومتكاثر"
}

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route("/", methods=["GET", "POST"])
def upload_file():
    result = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join("uploads", file.filename)
            file.save(filepath)
            img_array = preprocess_image(filepath)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            result = class_labels[predicted_class]
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)