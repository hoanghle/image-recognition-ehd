import numpy as np
from ehd import classes
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.preprocessing import StandardScaler

#đọc dữ liệu
ehd_features = np.load('ehd_features.npy')
labels = np.load('labels.npy')

print ("shape of ehd_features:", ehd_features.shape)
print("Shape of labels:", labels.shape)

# In số lượng ảnh của từng nhãn
unique, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique, counts):
    print(f"Class {classes[label]}: {count} images")
#print("label:", labels[3])

# In tên nhãn đầu tiên
#print("Label (class name):", classes[labels[0]])

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
ehd_features = scaler.fit_transform(ehd_features)

X_train, X_test, y_train, y_test = train_test_split(ehd_features, labels, test_size=0.2, random_state=42)

print("Shape of X_train:,", X_train.shape)
print("Shape of X_test:,", X_test.shape)

model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=classes))

# Lưu mô hình và scaler
joblib.dump(model, 'ehd_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Saved model and scaler to files.")

