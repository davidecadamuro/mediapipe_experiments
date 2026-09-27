import os
import cv2 as cv
import logging
import mediapipe as mp
import time

log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "landmark.log")
logging.basicConfig(
    filename=log_path,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'mp_models/hand_landmarker.task'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Python cannot find the file at: '{os.path.abspath(model_path)}'. Check your folder structure!")

with open(model_path, 'rb') as f:
    model_buffer = f.read()

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_buffer=model_buffer),
    running_mode=VisionRunningMode.VIDEO, num_hands=2, 
    min_hand_detection_confidence=0.5)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    cap.release()
    raise RuntimeError(
        "Could not open the camera. Grant camera access to the app running Python in "
        "System Settings > Privacy & Security > Camera, then try again."
    )

last_print_time = time.monotonic()
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            logging.warning("Could not read a frame from the camera.")
            break

        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        timestamp_ms = time.monotonic_ns() // 1_000_000
        results = landmarker.detect_for_video(image, timestamp_ms)

        current_time = time.monotonic()
        if current_time - last_print_time >= 1.0:
            if results.hand_landmarks:
                for hand_index, hand_landmarks in enumerate(results.hand_landmarks):
                    category = results.handedness[hand_index][0]
                    coordinates = ", ".join(
                        f"{index}:({landmark.x:.3f}, {landmark.y:.3f}, {landmark.z:.3f})"
                        for index, landmark in enumerate(hand_landmarks)
                    )
                    logging.info(
                        f"Hand {hand_index + 1} {category.category_name} "
                        f"({category.score:.2f}): {coordinates}"
                    )
            else:
                logging.info("No hands detected.")
            last_print_time = current_time

        for hand_landmarks in results.hand_landmarks:
            mp_drawing.draw_landmarks(
                rgb_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        annotated_frame = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)
        cv.imshow('MediaPipe Hands', cv.flip(annotated_frame, 1))
        if cv.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv.destroyAllWindows()
for _ in range(5):
    cv.waitKey(1)