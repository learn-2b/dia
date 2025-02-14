import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageTk

# تحميل النموذج المدرب
model = load_model("best_model.keras")

# إعداد أسماء الفئات
class_labels = {
    0: "No DR - العين سليمة",
    1: "Mild DR - اعتلال شبكية بسيط",
    2: "Moderate DR - اعتلال شبكية متوسط",
    3: "Severe DR - اعتلال شبكية شديد",
    4: "Proliferative DR - اعتلال شبكية خطير ومتكاثر"
}

# دالة لتحليل الصورة
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return class_labels[predicted_class]

# دالة اختيار الصورة
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # تصغير الصورة لعرضها
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk

        # التنبؤ بالحالة
        result = predict(file_path)
        result_label.config(text=f"Result: {result}")

# إنشاء واجهة المستخدم باستخدام Tkinter
window = tk.Tk()
window.title("Diabetic Retinopathy Classifier")

# مكونات الواجهة
upload_button = tk.Button(window, text="Upload Image", command=upload_image, font=("Arial", 14))
upload_button.pack(pady=20)

img_label = tk.Label(window)
img_label.pack(pady=10)

result_label = tk.Label(window, text="Result: ", font=("Arial", 16))
result_label.pack(pady=20)

# تشغيل التطبيق
window.mainloop()
