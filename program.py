import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Initialize list to store landmark coordinates
            landmark_list = []

            # Store landmark coordinates
            for id, lm in enumerate(hand_landmarks.landmark):
                # Get the coordinates
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([cx, cy])

            # Gesture recognition logic
            gesture = None
            if len(landmark_list) == 21:
                # Open Hand (Palm) Gesture
                if (landmark_list[4][1] > landmark_list[3][1] and
                        landmark_list[8][1] < landmark_list[6][1]):
                    gesture = "Hi"

                # Gesture 1: Thumbs Up
                if (landmark_list[4][1] < landmark_list[3][1] and
                        landmark_list[8][1] > landmark_list[6][1]):
                    gesture = "Thumbs Up"

                # Gesture 3: Fist
                if (landmark_list[4][1] < landmark_list[3][1] and
                        landmark_list[8][1] > landmark_list[6][1] and
                        landmark_list[12][1] > landmark_list[10][1]):
                    gesture = "Fist"

                # Gesture 4: Peace Sign
                if (landmark_list[8][1] < landmark_list[6][1] and
                        landmark_list[12][1] < landmark_list[10][1] and
                        landmark_list[16][1] > landmark_list[14][1]):
                    gesture = "Peace Sign"

                # Gesture 5: OK
                if (landmark_list[4][0] < landmark_list[3][0] and
                        landmark_list[8][0] < landmark_list[7][0] and
                        abs(landmark_list[4][1] - landmark_list[8][1]) < 20):
                    gesture = "OK"

            # Display the corresponding text
            if gesture:
                cv2.putText(frame, gesture, (landmark_list[0][0] - 50, landmark_list[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
