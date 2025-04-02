import numpy as np
from ehd import classes, rgb_to_gray, resize_image, find_ehd
import joblib
from PIL import Image

# Tải mô hình và scaler
model = joblib.load('ehd_model.pkl')
scaler = joblib.load('scaler.pkl')

# Hàm trích xuất EHD (tương tự trong ehd.py)
def extract_ehd(image_path):
    img_pil = Image.open(image_path)
    img_array = np.array(img_pil)
    img_gray = rgb_to_gray(img_array)
    img_gray = resize_image(img_gray, target_size=(256, 256))
    ehd = find_ehd(img_gray)
    return ehd

new_image_path = r'D:\Documents\tailieuhoctap\N4K2\Khaiphadldpt\Test images\skytest.jpg'
# Trích xuất EHD cho ảnh mới
new_ehd = extract_ehd(new_image_path)
new_ehd = new_ehd.reshape(1, -1)  # Reshape thành (1, 5)

# Chuẩn hóa EHD (dùng scaler đã lưu)
new_ehd = scaler.transform(new_ehd)

# Dự đoán
prediction = model.predict(new_ehd)
predicted_class = classes[prediction[0]]
print("Predicted class:", predicted_class)