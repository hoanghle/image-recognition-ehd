import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import os

dataset_path = r'D:\Documents\tailieuhoctap\N4K2\Khaiphadldpt\train_data'  # Dùng 'r' để tránh lỗi ký tự escape
classes = ['car', 'chimney', 'cow', 'door', 'plane', 'skyoi', 'tableware', 'tree', 'window']
labels =[]
ehd_features = []

#img_pil = Image.open(img_path)  # Đọc ảnh bằng PIL
#img_array = np.array(img_pil)  # Chuyển thành mảng NumPy (RGB)

#plt.imshow(img_pil)
#plt.title('original image')
#plt.axis('on')
#plt.show()



def rgb_to_gray(img):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def resize_image(img, target_size=(256, 256)):
    # Chuyển mảng NumPy thành PIL Image
    img_pil = Image.fromarray(img.astype(np.uint8))

    # Resize giữ tỷ lệ khung hình
    img_pil.thumbnail(target_size, Image.LANCZOS)

    # Tạo ảnh mới 256x256, màu đen (hoặc trắng), để padding
    new_img = Image.new('L', target_size, 0)  # 'L' cho ảnh grayscale, 0 là màu đen
    offset = ((target_size[0] - img_pil.size[0]) // 2,
              (target_size[1] - img_pil.size[1]) // 2)
    new_img.paste(img_pil, offset)

    # Chuyển lại thành mảng NumPy
    img_resized = np.array(new_img)
    return img_resized

# Chuyển sang grayscale
#img = rgb_to_gray(img_array)

# Resize ảnh
#img = resize_image(img, target_size=(256, 256))

# Hiển thị ảnh grayscale sau khi resize
#plt.imshow(img, cmap='gray')  # Dùng cmap='gray' để hiển thị đúng màu gray
#plt.title('gray image')
#plt.axis('on')
#plt.show()


def find_ehd(img):
    r, c = np.shape(img)  # Chỉ gán 2 biến vì img là 2D (grayscale)

    M = 4 * np.ceil(r / 4)  # ceil: làm tròn thành số nguyên
    N = 4 * np.ceil(c / 4)

    img = np.reshape(img, (int(M), int(N)))

    AllBins = np.zeros((17, 5))  # khởi tạo bin với 17 hàng 5 cột

    # Danh sách để lưu các khối
    blocks = []

    p = 1
    K = 0
    for i in range(4):  # Vòng lặp ngoài: Điều khiển hàng
        L = 0
        for j in range(4):  # Vòng lặp trong: Điều khiển cột
            block = img[K:K + int(M / 4), L:L + int(N / 4)]
            blocks.append(block)  # Lưu khối vào danh sách
            AllBins[p, :] = get_bins(np.double(block))
            L = L + int(N / 4)  # Tăng L để chuyển sang cột tiếp theo
            p = p + 1
        K = K + int(M / 4)  # Tăng K để chuyển sang hàng tiếp theo

    # Hiển thị các khối
    #fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    #fig.suptitle('16 Blocks of the Image', fontsize=16)

    #for idx, ax in enumerate(axes.flat):
    #    ax.imshow(blocks[idx], cmap='gray')
    #    ax.set_title(f'Block {idx + 1}')
    #    ax.axis('off')

    #plt.tight_layout()
    #plt.show()

    GlobalBin = np.mean(AllBins)
    AllBins[16, :] = np.round(GlobalBin)
    ehd = np.reshape(np.transpose(AllBins), [1, 85])
    ehd = ehd[0, -5:]

    # Vẽ biểu đồ cột cho EHD
    #edge_types = ['Vertical', 'Horizontal', 'Diagonal 45°', 'Diagonal 135°', 'Isotropic']
    #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Màu sắc khác nhau cho từng loại cạnh
    #plt.figure(figsize=(8, 5))
    #bars = plt.bar(edge_types, ehd, color=colors, edgecolor='black')

    # Thêm giá trị lên trên mỗi cột
    #for bar in bars:
     #   yval = bar.get_height()
     #   plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f'{yval:.1f}',
     #            ha='center', va='bottom', fontsize=10)

    #plt.title('Edge Histogram Descriptor (EHD)', fontsize=14, pad=20)
    #plt.xlabel('Edge Type', fontsize=12)
    #plt.ylabel('Value', fontsize=12)
    #plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    #plt.tight_layout()
    #plt.show()

    return ehd


def get_bins(imgb):
    M, N = imgb.shape
    M = 2 * np.ceil(M / 2)
    N = 2 * np.ceil(N / 2)

    #print(M)
    #print(N)
    imgb = np.reshape(imgb, (int(M), int(N)))

    bins = np.zeros((1, 5))

    V = np.array([[1, -1], [1, -1]])  # vertical
    H = np.array([[1, 1], [-1, -1]])  # horizontal
    D45 = np.array([[1.414, 0], [0, -1.414]])
    D135 = np.array([[0, 1.414], [-1.414, 0]])
    Isot = np.array([[2, -2], [-2, 2]])
    T = 50  # threshold

    nobr = int(M / 2)  # loop limits
    nobc = int(N / 2)  # loop limits
    L = 0

    for _ in range(nobc):
        K = 0
        for _ in range(nobr):
            block = imgb[K:K + 2, L:L + 2]  # Extracting 2x2 block
            pv = np.abs(np.sum(np.sum(block * V)))  # apply operators
            ph = np.abs(np.sum(np.sum(block * H)))
            pd45 = np.abs(np.sum(np.sum(block * D45)))
            pd135 = np.abs(np.sum(np.sum(block * D135)))
            pisot = np.abs(np.sum(np.sum(block * Isot)))
            parray = [pv, ph, pd45, pd135, pisot]
            index = np.argmax(parray)  # get the index of max value
            value = parray[index]  # get the max value
            if value >= T:
                bins[0, index] = bins[0, index] + 1  # update bins values
            K = K + 2
        L = L + 2
    return bins

def extract_ehd(image_path):
    img_pil = Image.open(image_path)
    img_array = np.array(img_pil)
    img_gray = rgb_to_gray(img_array)
    img_gray = resize_image(img_gray, target_size=(256, 256))  # Thêm resize
    ehd = find_ehd(img_gray)
    return ehd

# Duyệt qua từng lớp
for class_idx, class_name in enumerate(classes):
    class_path = os.path.join(dataset_path, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        try:
            # Trích xuất EHD
            ehd = extract_ehd(img_path)
            ehd_features.append(ehd)
            labels.append(class_idx)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Chuyển thành mảng NumPy
ehd_features = np.array(ehd_features)  # Shape: (1000, 5)
labels = np.array(labels)  # Shape: (1000,)

np.save('ehd_features.npy', ehd_features)
np.save('labels.npy', labels)

print("Saved ehd_features and labels to files.")
# Gọi hàm find_ehd
#ehd = find_ehd(img)
#print("EHD:", ehd)