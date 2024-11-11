import cv2
import mediapipe as mp


def main():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        max_num_hands=1)

    cap = cv2.VideoCapture(2)

    # Main Loop
    while cap.isOpened():

        # Read video frame by frame
        success, img = cap.read()

        img = cv2.resize(img, (640, 360))

        # Flip the image(frame)
        img = cv2.flip(img, 1)

        # Convert BGR image to RGB image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the RGB image
        results = hands.process(imgRGB)
        # If hands are present in image(frame)
        # print(results)
        if results.multi_hand_landmarks:
            for i in results.multi_hand_landmarks:
                # print(i)
                for j in i.landmark:
                    print(j)
                    cv2.circle(img, (int(j.x*640), int(j.y*360)), 5, (0, 255, 0), -1)


        # Display Video and when 'q' is entered, destroy the window
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

if __name__ == "__main__":
    main()
