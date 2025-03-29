from picamera2 import Picamera2
import cv2
import mediapipe as mp
import time

# Setup MediaPipe
drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands

# Initialize PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(2)  # Let the camera warm up

with handsModule.Hands(
    static_image_mode=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
) as hands:

    while True:
        frame = picam2.capture_array()

        # Resize just in case, though we configured 640x480
        frame1 = cv2.resize(frame, (640, 480))

        results = hands.process(cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR))  # PiCamera gives RGB, OpenCV needs BGR

        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame1, handLandmarks, handsModule.HAND_CONNECTIONS)

        cv2.imshow("PiCam + MediaPipe", frame1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
picam2.stop()
picam2.close()