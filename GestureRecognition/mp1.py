import cv2
import mediapipe as mp
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    return results

def detect_shrugging(results, shrug_times, current_time):
    if results.pose_landmarks:
        left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        
        if left_shoulder.y > 0.03 + right_shoulder.y:
            shrug_times.append(current_time)
            
        if right_shoulder.y > 0.03 + left_shoulder.y:
            shrug_times.append(current_time)

def detect_frowning(results, frown_times, current_time):
    if results.face_landmarks:
        left_eyebrow_inner = results.face_landmarks.landmark[65]  # Left eyebrow inner
        right_eyebrow_inner = results.face_landmarks.landmark[295]  # Right eyebrow inner
        left_eye_top = results.face_landmarks.landmark[159]  # Left eye top
        right_eye_top = results.face_landmarks.landmark[386]  # Right eye top

        left_eyebrow_distance = left_eye_top.y - left_eyebrow_inner.y
        right_eyebrow_distance = right_eye_top.y - right_eyebrow_inner.y 

        if left_eyebrow_distance < 0.026 and right_eyebrow_distance < 0.026:
            frown_times.append(current_time)

cap = cv2.VideoCapture(0)

shrug_times = []
frown_times = []

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        results = mediapipe_detection(frame, holistic)

        detect_shrugging(results, shrug_times, current_time)
        detect_frowning(results, frown_times, current_time)

        shrug_times = [t for t in shrug_times if current_time - t <= 20]
        frown_times = [t for t in frown_times if current_time - t <= 20]

        if len(shrug_times) > 0 and len(frown_times) > 0:
            status_text = 'Slightly Suspicious Status 2/2'
            cv2.putText(frame, status_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            if results.pose_landmarks:
                left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

                if left_shoulder.y > 0.03 + right_shoulder.y:
                    cv2.putText(frame, 'Sol omuz silkme', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if right_shoulder.y > 0.03 + left_shoulder.y:
                    cv2.putText(frame, 'Sag omuz silkme', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if results.face_landmarks:
                left_eyebrow_inner = results.face_landmarks.landmark[65]  # Left eyebrow inner
                right_eyebrow_inner = results.face_landmarks.landmark[295]  # Right eyebrow inner
                left_eye_top = results.face_landmarks.landmark[159]  # Left eye top
                right_eye_top = results.face_landmarks.landmark[386]  # Right eye top

                left_eyebrow_distance = left_eye_top.y - left_eyebrow_inner.y
                right_eyebrow_distance = right_eye_top.y - right_eyebrow_inner.y 

                if left_eyebrow_distance < 0.026 and right_eyebrow_distance < 0.026:
                    cv2.putText(frame, 'Kas catma', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        if results.face_landmarks:
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                      landmark_drawing_spec=None, connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow("OpenCV Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
