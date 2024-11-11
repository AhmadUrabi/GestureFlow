import cv2
import mediapipe as mp
import pyautogui  
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    prev_x = 0  
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                current_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x

                if current_x - prev_x > 0.1:
                    print("Swipe Right Detected! -> Next Slide")
                    pyautogui.press('right')  

                elif prev_x - current_x > 0.1:
                    print("Swipe Left Detected! -> Previous Slide")
                    pyautogui.press('left')  
                prev_x = current_x

        cv2.imshow('Gesture-Controlled Slides', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
