import cv2
import mediapipe as mp
import csv

def write_frame_entry(landmarks, count):
    with open('hand_landmarks.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        result_row = [f'Frame {count}']
        for landmark in landmarks.landmark:
            result_row.append(landmark.x)
            result_row.append(landmark.y)
            result_row.append(landmark.z)
        writer.writerow(result_row)


def process_video(video_file):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        max_num_hands=1)

    cap = cv2.VideoCapture(video_file)
    count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        process_frame(frame, hands, count)
        count += 1

        cv2.imshow('window-name', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame, hands,count):
    img = cv2.resize(frame, (640, 480))

    # Flip the image(frame)
    img = cv2.flip(img, 1)

    # Convert BGR image to RGB image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Write landmarks to CSV
            write_frame_entry(hand_landmarks, count)

            # Draw circles for each landmark
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * 640)
                y = int(landmark.y * 480)
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        # Save annotated image
        cv2.imwrite(f"frames/frame{count}.jpg", img)

def main():
    process_video('test.mkv')

if __name__ == "__main__":
    main()
