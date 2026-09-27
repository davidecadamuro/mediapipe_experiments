import os
import argparse
import cv2 as cv
import logging
import mediapipe as mp
import sys
import time

def get_log_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="Save logs to ./data/<filename>.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.filename:
        data_dir = os.path.join(script_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        log_path = os.path.join(data_dir, args.filename)
        if os.path.exists(log_path):
            raise FileExistsError(
                f"The file '{args.filename}' already exists in the data folder."
            )
        return log_path

    return os.path.join(script_dir, "landmark.log")


def draw_log_counter(display_frame, collected_logs):
    counter_text = f"Logs: {collected_logs}/100"
    text_size = cv.getTextSize(
        counter_text, cv.FONT_HERSHEY_SIMPLEX, 1, 2
    )[0]
    text_position = (display_frame.shape[1] - text_size[0] - 10, 35)
    cv.putText(
        display_frame,
        counter_text,
        text_position,
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        display_frame,
        counter_text,
        text_position,
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (88, 205, 54),
        2,
        cv.LINE_AA,
    )
    return display_frame


try:
    log_path = get_log_path()
except FileExistsError as error:
    print(f"Error: {error}", file=sys.stderr)
    raise SystemExit(1) from error

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

collected_logs = 0
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

        for hand_landmarks in results.hand_landmarks:
            mp_drawing.draw_landmarks(
                rgb_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        annotated_frame = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)
        display_frame = cv.flip(annotated_frame, 1)
        display_frame = draw_log_counter(display_frame, collected_logs)
        cv.imshow('MediaPipe Hands', display_frame)
        key = cv.waitKey(5) & 0xFF
        if key == ord(' '):
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
            collected_logs += 1

        if key == 27 or collected_logs >= 100:
            break
cap.release()
cv.destroyAllWindows()
for _ in range(5):
    cv.waitKey(1)