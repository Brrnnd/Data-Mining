import os
from torchvision import transforms
from PIL import Image

# Augmentasi menggunakan transformasi torchvision
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

# Membaca gambar
image_path = r'D:\KULIAH\DATA MINING\PROJECT DATA MINING\test\dataset\train\yesterday\yesterday_10_jpg.rf.f1c70f13bda895c0a0c64f9c297da3cf.jpg'
image = Image.open(image_path)

# Menentukan direktori penyimpanan untuk gambar hasil augmentasi
output_directory = r'D:\KULIAH\DATA MINING\PROJECT DATA MINING\test\dataset\train\yesterday'
os.makedirs(output_directory, exist_ok=True)  # Membuat direktori jika belum ada

# Jumlah augmentasi yang diinginkan
num_augmentations = 500


for i in range(num_augmentations):
    # Augmentasi
    augmented_image = transform(image)
    
    # Menyimpan gambar hasil augmentasi
    augmented_image_pil = transforms.ToPILImage()(augmented_image)
    output_path = os.path.join(output_directory, f'yesterday{i + 1}.jpg')
    augmented_image_pil.save(output_path)

    print(f'Gambar hasil augmentasi {i + 1} disimpan di: {output_path}')
