import csv
import pickle
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prepare_data import normalize_landmarks


PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODELS_DIR / "hare_hunter.cpickle"
LEADERBOARD_PATH = PROJECT_DIR / "leaderboard.csv"
HAND_MODEL_PATH = PROJECT_DIR / "mp_models" / "hand_landmarker.task"
ARCADE_FONT_PATH = PROJECT_DIR / "fonts" / "PressStart2P-Regular.ttf"
GAME_DURATION_SECONDS = 30.0
LEADERBOARD_LIMIT = 20
WINDOW_NAME = "Hare Hunter"
ARCADE_COLOR = (255, 255, 255)
ARCADE_OUTLINE_COLOR = (0, 0, 0)


@lru_cache(maxsize=8)
def get_arcade_font(size: int):
    """Load and cache the local Press Start 2P font at the requested size."""
    if not ARCADE_FONT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing game font: {ARCADE_FONT_PATH}. "
            "Place PressStart2P-Regular.ttf in ./fonts."
        )
    return ImageFont.truetype(str(ARCADE_FONT_PATH), size)


def get_arcade_text_size(text: str, scale: float, thickness: int) -> tuple[int, int]:
    """Return the rendered size of arcade text, including its outline."""
    font = get_arcade_font(max(10, int(36 * scale)))
    left, top, right, bottom = font.getbbox(
        text,
        stroke_width=thickness + 2,
    )
    return right - left, bottom - top


def draw_arcade_text(
    display_frame,
    text: str,
    position: tuple[int, int],
    scale: float = 0.8,
    thickness: int = 2,
) -> None:
    """Draw Press Start 2P text with a dark outline."""
    frame_rgb = cv.cvtColor(display_frame, cv.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image)
    font = get_arcade_font(max(10, int(36 * scale)))
    draw.text(
        position,
        text,
        font=font,
        fill=ARCADE_COLOR,
        stroke_width=thickness + 2,
        stroke_fill=ARCADE_OUTLINE_COLOR,
    )
    display_frame[:] = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)


@dataclass
class GameState:
    started_at: float | None = None
    score: int = 0
    finished: bool = False
    last_assignment: dict[str, str] | None = None
    score_flash_until: float = 0.0

    @property
    def started(self) -> bool:
        return self.started_at is not None


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


def classify_hands(model, results) -> list[tuple[object, str, float, str]]:
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
        predictions.append((hand_landmarks, label, confidence, handedness))

    return predictions


def draw_predictions(display_frame, predictions) -> None:
    """Draw predicted labels next to each detected hand."""
    frame_height, frame_width = display_frame.shape[:2]
    for hand_landmarks, label, confidence, _ in predictions:
        wrist = hand_landmarks[0]
        text = f"{label}" #f"{label} ({confidence:.2f})"
        text_size = get_arcade_text_size(text, 0.8, 2)
        text_x = int((1.0 - wrist.x) * frame_width) - text_size[0] // 2
        text_y = int(wrist.y * frame_height) - 12
        text_x = max(5, min(text_x, frame_width - text_size[0] - 5))
        text_y = max(text_size[1] + 5, text_y)
        text_position = (text_x, text_y)

        draw_arcade_text(display_frame, text, text_position)


def draw_score_glow(display_frame, predictions, current_time: float, game_state: GameState) -> None:
    """Draw a brief glow around every landmark after a point is scored."""
    if current_time >= game_state.score_flash_until:
        return

    frame_height, frame_width = display_frame.shape[:2]
    glow = np.zeros_like(display_frame)
    for hand_landmarks, _, _, _ in predictions:
        for landmark in hand_landmarks:
            position = (
                int((1.0 - landmark.x) * frame_width),
                int(landmark.y * frame_height),
            )
            cv.circle(glow, position, 10, (255, 255, 255), -1, cv.LINE_AA)

    blurred_glow = cv.GaussianBlur(glow, (0, 0), 12)
    display_frame[:] = cv.addWeighted(display_frame, 1.0, blurred_glow, 0.8, 0)
    display_frame[:] = cv.addWeighted(display_frame, 1.0, glow, 0.35, 0)


def has_starting_gestures(predictions) -> bool:
    """Return whether both hare and hunter are currently detected."""
    assignment = get_gesture_assignment(predictions)
    return assignment is not None


def get_gesture_assignment(predictions) -> dict[str, str] | None:
    """Return the complete left/right hare-hunter assignment, if present."""
    assignment = {
        handedness: label
        for _, label, _, handedness in predictions
        if handedness in {"Left", "Right"} and label in {"hare", "hunter"}
    }
    if set(assignment) != {"Left", "Right"}:
        return None
    if set(assignment.values()) != {"hare", "hunter"}:
        return None
    return assignment


def update_game_state(game_state: GameState, predictions, current_time: float) -> None:
    """Start the game and finish it when its time limit expires."""
    assignment = get_gesture_assignment(predictions)
    if not game_state.started and assignment is not None:
        game_state.started_at = current_time
        game_state.last_assignment = assignment

    elif game_state.started and assignment is not None:
        if game_state.last_assignment != assignment:
            game_state.score += 1
            game_state.last_assignment = assignment
            game_state.score_flash_until = current_time + 0.2

    if game_state.started and current_time - game_state.started_at >= GAME_DURATION_SECONDS:
        game_state.finished = True


def draw_game_status(display_frame, game_state: GameState, current_time: float) -> None:
    """Draw the game's current status and score."""
    if game_state.finished:
        status_text = f"Game over - Score: {game_state.score}"
    elif not game_state.started:
        status_text = "Show hare and hunter to start"
    else:
        elapsed = current_time - game_state.started_at
        remaining = max(0.0, GAME_DURATION_SECONDS - elapsed)
        status_text = f"Time: {remaining:.1f}s  Score: {game_state.score}"

    draw_arcade_text(display_frame, status_text, (20, 40))


def draw_centered_text(display_frame, text: str, y: int) -> None:
    """Draw outlined text centered horizontally on the frame."""
    frame_width = display_frame.shape[1]
    text_size = get_arcade_text_size(text, 0.8, 2)
    position = ((frame_width - text_size[0]) // 2, y)
    draw_arcade_text(display_frame, text, position)


def draw_leaderboard(
    display_frame,
    entries: list[dict[str, str]],
    top_y: int = 70,
) -> None:
    """Draw the current top scores starting at the requested vertical position."""
    frame_width = display_frame.shape[1]
    title = "LEADERBOARD"
    title_width, _ = get_arcade_text_size(title, 0.6, 2)
    title_x = max(5, (frame_width - title_width) // 2)
    draw_arcade_text(display_frame, title, (title_x, top_y), 0.6)
    for rank, entry in enumerate(entries[:LEADERBOARD_LIMIT], start=1):
        row = f"{rank:2}. {entry['name'][:18]:18} {entry['score']}"
        row_width, _ = get_arcade_text_size(row, 0.4, 1)
        row_x = max(5, (frame_width - row_width) // 2)
        draw_arcade_text(display_frame, row, (row_x, top_y + rank * 20), 0.4, 1)


def get_player_name(
    display_frame,
    score: int,
    leaderboard: list[dict[str, str]],
) -> str | None:
    """Prompt for a player name and return it after Enter is pressed."""
    player_name = ""
    prompt_frame = display_frame.copy()

    while True:
        frame = prompt_frame.copy()
        draw_centered_text(frame, f"Final score: {score}", 90)
        draw_centered_text(frame, "Enter your name, then press Enter", 135)
        draw_centered_text(frame, f"> {player_name}_", 185)
        draw_leaderboard(frame, leaderboard, top_y=250)
        cv.imshow(WINDOW_NAME, frame)

        key = cv.waitKeyEx(50)
        key_code = key & 0xFF
        if key_code in (10, 13) and player_name:
            return player_name
        if key_code == 27:
            return None
        if key_code in (8, 127):
            player_name = player_name[:-1]
        elif 32 <= key_code <= 126 and len(player_name) < 24:
            player_name += chr(key_code)


def load_leaderboard() -> list[dict[str, str]]:
    """Load valid leaderboard entries from disk."""
    if not LEADERBOARD_PATH.exists():
        return []

    with LEADERBOARD_PATH.open(newline="", encoding="utf-8") as leaderboard_file:
        entries = []
        for entry in csv.DictReader(leaderboard_file):
            try:
                int(entry["score"])
            except (KeyError, TypeError, ValueError):
                continue
            entries.append(entry)
    return entries


def qualifies_for_leaderboard(score: int) -> bool:
    """Return whether a score belongs in the top twenty."""
    entries = load_leaderboard()
    if len(entries) < LEADERBOARD_LIMIT:
        return True
    twentieth_score = min(int(entry["score"]) for entry in entries)
    return score >= twentieth_score


def save_leaderboard_entry(player_name: str, score: int) -> None:
    """Append a completed game to the leaderboard CSV."""
    entries = load_leaderboard()
    entries.append(
        {
            "name": player_name,
            "score": str(score),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    entries.sort(key=lambda entry: int(entry["score"]), reverse=True)

    with LEADERBOARD_PATH.open("w", newline="", encoding="utf-8") as leaderboard_file:
        writer = csv.DictWriter(
            leaderboard_file,
            fieldnames=("name", "score", "timestamp"),
        )
        writer.writeheader()
        writer.writerows(entries[:LEADERBOARD_LIMIT])


def wait_for_exit(
    display_frame,
    message: str,
    leaderboard: list[dict[str, str]],
) -> None:
    """Keep a final message visible until Escape is pressed."""
    while True:
        frame = display_frame.copy()
        draw_centered_text(frame, message, 135)
        draw_centered_text(frame, "Press Escape to exit", 185)
        draw_leaderboard(frame, leaderboard, top_y=250)
        cv.imshow(WINDOW_NAME, frame)
        if cv.waitKey(50) & 0xFF == 27:
            return


def run_camera(model) -> None:
    """Run the gesture game until its timer ends or Escape is pressed."""
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
        game_state = GameState()
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
                current_time = time.monotonic()
                update_game_state(game_state, predictions, current_time)

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
                draw_score_glow(display_frame, predictions, current_time, game_state)
                draw_predictions(display_frame, predictions)
                draw_game_status(display_frame, game_state, current_time)
                cv.imshow(WINDOW_NAME, display_frame)
                if cv.waitKey(5) & 0xFF == 27:
                    break

                if game_state.finished:
                    leaderboard = load_leaderboard()
                    if qualifies_for_leaderboard(game_state.score):
                        player_name = get_player_name(
                            display_frame,
                            game_state.score,
                            leaderboard,
                        )
                        if player_name is not None:
                            save_leaderboard_entry(player_name, game_state.score)
                            game_state = GameState()
                            continue
                        else:
                            final_message = "Score not saved"
                        wait_for_exit(display_frame, final_message, leaderboard)
                    else:
                        wait_for_exit(
                            display_frame,
                            "Score not in top 20",
                            leaderboard,
                        )
                    game_state = GameState()
    finally:
        cap.release()
        cv.destroyAllWindows()


def main() -> None:
    """Run gesture recognition from the laptop camera."""
    model = load_model(MODEL_PATH)
    run_camera(model)


if __name__ == "__main__":
    main()