import torch
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib
import numpy as np
import cv2
import time

# YOLOv5 modelini yükleyin
model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5/runs/train/exp4/weights/last.pt', force_reload=True)

# Göz kırpma sayısını ve ortalama hesaplamak için değişkenler
blink_counter = 0
start_time = time.time()
blink_times = []

# Boyun dokunma sayısını tutmak için değişkenler
neck_touch_counter = 0
neck_touch_times = []

# Önceki karede gözlerin ve boynun durumu
previous_eye_closed = False
previous_neck_touch = False

# VideoCapture ile kamerayı başlatın
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Modeli kullanarak çerçeveyi analiz edin
    results = model(frame)

    # Sonuçları alın ve gözlerin durumunu kontrol edin
    eye_closed = False
    neck_touch_detected = False
    for det in results.xyxy[0]:  # results.xyxy[0] içindeki her bir tespit için
        if det[5] == 17:  # Örneğin, gözlerin kapalı olduğunu belirten sınıf ID'si 17 ise
            eye_closed = True
        if det[5] == 15:  # Boyun dokunma sınıf ID'si 18 ise (bunu modelinize göre değiştirin)
            neck_touch_detected = True

    # Gözlerin kapalı olduğunu tespit edersek ve önceki karede gözler açık ise sayaç artırılır
    if eye_closed and not previous_eye_closed:
        blink_counter += 1
        current_time = time.time() - start_time
        blink_times.append(current_time)

    # Boyun dokunma tespit edilirse ve önceki karede boyun dokunma yoksa sayaç artırılır
    current_time = time.time()
    if neck_touch_detected and not previous_neck_touch:
        neck_touch_counter += 1
        neck_touch_times.append(current_time)

    # Önceki karedeki durumu güncelle
    previous_eye_closed = eye_closed
    previous_neck_touch = neck_touch_detected

    # Boyun dokunma sayısını son 20 saniyeye göre güncelle
    neck_touch_times = [t for t in neck_touch_times if current_time - t <= 20]
    recent_neck_touch_counter = len(neck_touch_times)

    # Ortalama göz kırpma hızını hesaplayın
    elapsed_time = current_time - start_time
    if elapsed_time > 0:
        average_blink_rate = blink_counter / elapsed_time
    else:
        average_blink_rate = 0

    # Boyun dokunma ve göz kırpma durumunu kontrol edin ve ekrana yazdırın
    if recent_neck_touch_counter > 1 and average_blink_rate > 0.4826:
        cv2.putText(frame, 'Highly Suspicious Status 2/2', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        if average_blink_rate < 0.4826:
            cv2.putText(frame, f'Avg Blinks/Sec: {average_blink_rate:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Suspicious Avg Blink Value', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Neck Touch Count (last 20 sec): {recent_neck_touch_counter}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Sonuçları göster
    cv2.imshow('YOLO', np.squeeze(results.render()))

    # 'q' tuşuna basıldığında döngüden çıkın
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakın
cap.release()
cv2.destroyAllWindows()
