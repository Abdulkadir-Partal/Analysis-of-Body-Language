import torch
import cv2
import mediapipe as mp
import time
import numpy as np

# YOLOv5 modelini yükleyin
model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5/runs/train/exp4/weights/last.pt', force_reload=True)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Sayaçları başlatan bir fonksiyon
def reset_counters():
    global blink_counter, blink_times, neck_touch_counter, neck_touch_times, shrug_times, frown_times, start_time
    blink_counter = 0
    start_time = time.time()
    blink_times = []
    neck_touch_counter = 0
    neck_touch_times = []
    shrug_times = []
    frown_times = []

# Sayaçları başlat
reset_counters()

# Önceki karede gözlerin ve boynun durumu
previous_eye_closed = False
previous_neck_touch = False

# VideoCapture ile kamerayı başlatın
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # YOLOv5 modelini kullanarak çerçeveyi analiz edin
        results_yolo = model(frame)

        # MediaPipe Holistic modelini kullanarak çerçeveyi analiz edin
        results_mediapipe = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # YOLOv5 sonuçlarını alın ve gözlerin durumunu kontrol edin
        eye_closed = False
        neck_touch_detected = False
        for det in results_yolo.xyxy[0]:  # results.xyxy[0] içindeki her bir tespit için
            if det[5] == 17:  # Örneğin, gözlerin kapalı olduğunu belirten sınıf ID'si 17 ise
                eye_closed = True
            if det[5] == 15:  # Boyun dokunma sınıf ID'si 15 ise (bunu modelinize göre değiştirin)
                neck_touch_detected = True

        # Gözlerin kapalı olduğunu tespit edersek ve önceki karede gözler açık ise sayaç artırılır
        if eye_closed and not previous_eye_closed:
            blink_counter += 1
            current_time_blink = time.time() - start_time
            blink_times.append(current_time_blink)

        # Boyun dokunma tespit edilirse ve önceki karede boyun dokunma yoksa sayaç artırılır
        current_time_neck = time.time()
        if neck_touch_detected and not previous_neck_touch:
            neck_touch_counter += 1
            neck_touch_times.append(current_time_neck)

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

        if recent_neck_touch_counter > 1 and average_blink_rate > 0.4826:
            cv2.putText(frame, 'Highly Suspicious Status 2/2', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            if average_blink_rate < 0.4826:
                cv2.putText(frame, f'Avg Blinks/Sec: {average_blink_rate:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'Suspicious Avg Blink Value', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Neck Touch Count (last 20 sec): {recent_neck_touch_counter}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


        # MediaPipe sonuçlarını işleyin ve omuz silkme ve kaş çatma durumlarını kontrol edin
        if results_mediapipe.pose_landmarks:
            left_shoulder = results_mediapipe.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results_mediapipe.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

            if left_shoulder.y > 0.03 + right_shoulder.y:
                shrug_times.append(current_time)

            if right_shoulder.y > 0.03 + left_shoulder.y:
                shrug_times.append(current_time)

        if results_mediapipe.face_landmarks:
            left_eyebrow_inner = results_mediapipe.face_landmarks.landmark[65]  # Left eyebrow inner
            right_eyebrow_inner = results_mediapipe.face_landmarks.landmark[295]  # Right eyebrow inner
            left_eye_top = results_mediapipe.face_landmarks.landmark[159]  # Left eye top
            right_eye_top = results_mediapipe.face_landmarks.landmark[386]  # Right eye top

            left_eyebrow_distance = left_eye_top.y - left_eyebrow_inner.y
            right_eyebrow_distance = right_eye_top.y - right_eyebrow_inner.y

            if left_eyebrow_distance < 0.026 and right_eyebrow_distance < 0.026:
                frown_times.append(current_time)

        # Omuz silkme ve kaş çatma durumlarını son 20 saniyeye göre güncelle
        shrug_times = [t for t in shrug_times if current_time - t <= 20]
        frown_times = [t for t in frown_times if current_time - t <= 20]

        # Durumu kontrol edin ve ekrana yazdırın
        if len(shrug_times) > 0 and len(frown_times) > 0:
            status_text = 'Slightly Suspicious Status 2/2'
            cv2.putText(frame, status_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            #if average_blink_rate < 0.4826:
            #    cv2.putText(frame, f'Avg Blinks/Sec: {average_blink_rate:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            #else:
            #    cv2.putText(frame, 'Suspicious Avg Blink Value', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            #cv2.putText(frame, f'Neck Touch Count (last 20 sec): {recent_neck_touch_counter}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            if results_mediapipe.pose_landmarks:
                if left_shoulder.y > 0.03 + right_shoulder.y:
                    cv2.putText(frame, 'Shrug on one side', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if right_shoulder.y > 0.03 + left_shoulder.y:
                    cv2.putText(frame, 'Shrug on one side', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if results_mediapipe.face_landmarks:
                if left_eyebrow_distance < 0.026 and right_eyebrow_distance < 0.026:
                    cv2.putText(frame, 'Frowning', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # YOLOv5 ve MediaPipe sonuçlarını çizim
        frame = np.squeeze(results_yolo.render())  # YOLOv5 sonuçlarını çizim
        if results_mediapipe.face_landmarks:
            mp_drawing.draw_landmarks(frame, results_mediapipe.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                      landmark_drawing_spec=None, connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
        if results_mediapipe.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results_mediapipe.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results_mediapipe.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results_mediapipe.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results_mediapipe.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results_mediapipe.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Sonuçları göster
        cv2.imshow('Integrated YOLOv5 and MediaPipe', frame)

        # 'q' tuşuna basıldığında döngüden çıkın
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # 'r' tuşuna basıldığında sayaçları sıfırlayın
        if cv2.waitKey(10) & 0xFF == ord('r'):
            reset_counters()

# Kaynakları serbest bırakın
cap.release()
cv2.destroyAllWindows()
