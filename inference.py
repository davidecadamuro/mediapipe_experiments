import argparse
import pickle
import time
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np

from prepare_data import normalize_landmarks


PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
HAND_MODEL_PATH = PROJECT_DIR / "mp_models" / "hand_landmarker.task"


def get_model_path() -> Path:
    """Return the selected model path from the models directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="Model filename from the models directory.",
    )
    args = parser.parse_args()

    filename = Path(args.filename)
    if filename.name != args.filename:
        parser.error("filename must not contain a directory path")
    if filename.suffix == "":
        filename = filename.with_suffix(".cpickle")

    model_path = MODELS_DIR / filename.name
    if not model_path.is_file():
        parser.error(f"model file does not exist: {model_path}")

    return model_path


def load_model(model_path: Path):
    """Load a serialized scikit-learn model."""
    with model_path.open("rb") as model_file:
        return pickle.load(model_file)


def prepare_hand_features(hand_landmarks, handedness: str) -> np.ndarray:
    """Convert MediaPipe landmarks into the model's feature vector."""
    coordinates = [
        (landmark.x, landmark.y, landmark.z)
        for landmark in hand_landmarks
    ]
    return np.asarray(
        normalize_landmarks(coordinates, handedness),
        dtype=float,
    )


def classify_hands(model, results) -> list[tuple[object, str, float]]:
    """Prepare and classify every detected hand."""
    predictions = []
    for hand_index, hand_landmarks in enumerate(results.hand_landmarks):
        handedness = results.handedness[hand_index][0].category_name
        try:
            features = prepare_hand_features(hand_landmarks, handedness)
        except ValueError:
            continue

        probabilities = model.predict_proba([features])[0]
        class_index = int(np.argmax(probabilities))
        label = str(model.classes_[class_index])
        confidence = float(probabilities[class_index])
        predictions.append((hand_landmarks, label, confidence))

    return predictions


def draw_predictions(display_frame, predictions) -> None:
    """Draw predicted labels next to each detected hand."""
    frame_height, frame_width = display_frame.shape[:2]
    for hand_landmarks, label, confidence in predictions:
        wrist = hand_landmarks[0]
        text = f"{label} ({confidence:.2f})"
        text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = int((1.0 - wrist.x) * frame_width) - text_size[0] // 2
        text_y = int(wrist.y * frame_height) - 12
        text_x = max(5, min(text_x, frame_width - text_size[0] - 5))
        text_y = max(text_size[1] + 5, text_y)
        text_position = (text_x, text_y)

        cv.putText(
            display_frame,
            text,
            text_position,
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv.LINE_AA,
        )
        cv.putText(
            display_frame,
            text,
            text_position,
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (88, 205, 54),
            2,
            cv.LINE_AA,
        )


def run_camera(model) -> None:
    """Capture frames and detect hand landmarks until Escape is pressed."""
    mp_hands = mp.tasks.vision.HandLandmarksConnections
    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles

    with HAND_MODEL_PATH.open("rb") as model_file:
        model_buffer = model_file.read()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=model_buffer),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "Could not open the camera. Grant camera access to the app running "
            "Python in System Settings > Privacy & Security > Camera."
        )

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_image,
                )
                timestamp_ms = time.monotonic_ns() // 1_000_000
                results = landmarker.detect_for_video(image, timestamp_ms)
                predictions = classify_hands(model, results)

                for hand_landmarks in results.hand_landmarks:
                    mp_drawing.draw_landmarks(
                        rgb_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                display_frame = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)
                display_frame = cv.flip(display_frame, 1)
                draw_predictions(display_frame, predictions)
                cv.imshow("Gesture Recognition", display_frame)
                if cv.waitKey(5) & 0xFF == 27:
                    break
    finally:
        cap.release()
        cv.destroyAllWindows()


def main() -> None:
    """Run gesture recognition from the laptop camera."""
    model_path = get_model_path()
    model = load_model(model_path)
    run_camera(model)


if __name__ == "__main__":
    main()