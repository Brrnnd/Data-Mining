import torch
import cv2
import numpy as np
from torchvision import transforms
import os
from flask import Flask, render_template, Response  # Tambahkan render_template di sini

# Deteksi perangkat (GPU atau CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Cek keberadaan model
file_path = 'sign_language_words.pth'
if not os.path.exists(file_path):
    print("Model file does not exist. Check the path.")
    exit()

# Transformasi input dari kamera
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Definisikan ulang model
class SignLanguageModel(torch.nn.Module):
    def __init__(self):
        super(SignLanguageModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 16 * 16, 128)
        self.fc2 = torch.nn.Linear(128, 11)  # 11 kelas untuk kata-kata

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model dan set ke mode evaluasi
model = SignLanguageModel().to(device)
model.load_state_dict(torch.load(file_path), strict=False)
model.eval()

# Daftar kelas
classes = ['head', 'inside', 'leg', 'medicine', 'sick', 'snot', 'surgery', 'today', 'when', 'yes', 'yesterday']

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Variabel untuk menyimpan prediksi terakhir
last_prediction = None
sentence = ""
prediction = "Tidak ada prediksi"

# Load Haar Cascade untuk deteksi wajah dan tangan
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
hand_cascade_path = 'haarcascade_hand.xml'  # Ganti dengan jalur yang sesuai
hand_cascade = cv2.CascadeClassifier(hand_cascade_path)

if hand_cascade.empty():
    print("Error: Could not load hand cascade classifier.")
    exit()

def generate_frames():
    global last_prediction, sentence, prediction
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi wajah
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Deteksi tangan
        hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Gambarkan kotak di sekitar wajah yang terdeteksi
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Gambarkan kotak di sekitar tangan yang terdeteksi dan prediksi
        for (x, y, w, h) in hands:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Preproses area tangan untuk model bahasa isyarat
            hand_roi = frame[y:y + h, x:x + w]
            hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
            input_frame = transform(hand_roi_rgb).unsqueeze(0).to(device)

            # Prediksi
            try:
                with torch.no_grad():
                    outputs = model(input_frame)
                    _, predicted = torch.max(outputs, 1)
                    prediction = classes[predicted.item()]
            except Exception as e:
                print(f"Error during prediction: {e}")

            # Tampilkan prediksi jika berbeda dari prediksi sebelumnya
            if prediction != last_prediction:
                sentence += prediction + " "
                last_prediction = prediction

        # Tampilkan kalimat dan prediksi pada jendela video
        cv2.putText(frame, f'Kalimat: {sentence}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f'Prediksi: {prediction}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode frame ke JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Kirim frame sebagai stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
