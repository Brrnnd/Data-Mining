import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os

# Deteksi perangkat (GPU atau CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Transformasi dataset (resize dan normalisasi)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset bahasa isyarat
train_data = datasets.ImageFolder('dataset/train', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Load dataset pengujian
test_data = datasets.ImageFolder('dataset/test', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Definisikan CNN
class SignLanguageModel(nn.Module):
    def __init__(self):
        super(SignLanguageModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 11)  # 11 kelas untuk kata-kata

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Inisialisasi model, loss function, dan optimizer
model = SignLanguageModel().to(device)  # Pindahkan model ke GPU/CPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    start_time = time.time()
    running_loss = 0.0
    total_correct = 0  # Inisialisasi untuk menghitung akurasi
    total_samples = 0   # Inisialisasi total samples

    for images, labels in train_loader:
        # Pindahkan data ke GPU/CPU
        images, labels = images.to(device), labels.to(device)

        # Reset gradien
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass dan optimisasi
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Hitung akurasi
        _, predicted = torch.max(outputs.data, 1)  # Dapatkan prediksi
        total_samples += labels.size(0)  # Jumlah total samples dalam batch
        total_correct += (predicted == labels).sum().item()  # Hitung prediksi yang benar

    epoch_time = time.time() - start_time
    accuracy = total_correct / total_samples * 100  # Hitung akurasi dalam persen
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, '
          f'Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f} seconds')

# Daftar label sesuai dataset
labels_list = ['head', 'inside', 'leg', 'medicine', 'sick', 'snot', 'surgery', 'today', 'when', 'yes', 'yesterday']

# Testing loop
model.eval()  # Ubah model ke mode evaluasi
test_correct = 0
test_total = 0

# Inisialisasi penghitung untuk setiap label
label_correct = [0] * len(labels_list)
label_total = [0] * len(labels_list)

with torch.no_grad():  # Non-aktifkan gradien untuk pengujian
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        # Hitung benar dan total per label
        for label, pred in zip(labels, predicted):
            label_total[label.item()] += 1
            if label == pred:
                label_correct[label.item()] += 1

# Hitung akurasi keseluruhan
test_accuracy = test_correct / test_total * 100  # Hitung akurasi pengujian
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Hitung dan tampilkan akurasi per label
print("\nAccuracy per label:")
for i, label in enumerate(labels_list):
    if label_total[i] > 0:
        label_accuracy = label_correct[i] / label_total[i] * 100
        print(f'{label}: {label_accuracy:.2f}% ({label_correct[i]}/{label_total[i]})')
    else:
        print(f'{label}: No samples')


# Buat direktori models jika belum ada
if not os.path.exists('models'):
    os.makedirs('models')

# Simpan model
torch.save(model.state_dict(), 'models/sign_language_words.pth')
print("Model saved!")
