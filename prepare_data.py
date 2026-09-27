import csv
import math
from pathlib import Path
import re


DATA_DIR = Path(__file__).parent / "data"
HAND_LINE_PATTERN = re.compile(r"\b(Left|Right)\s+\([^)]*\):\s*(.*)$")
LANDMARK_PATTERN = re.compile(
    r"\d+:\(\s*([-+]?\d*\.?\d+)\s*,\s*"
    r"([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)"
)


def load_labeled_files(data_dir: Path = DATA_DIR) -> list[tuple[Path, str]]:
    """Return each data file paired with its gesture label."""
    labeled_files = []
    for file_path in sorted(data_dir.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() != ".log":
            continue

        filename = file_path.stem
        base_name, separator, suffix = filename.rpartition("_")
        label = base_name if separator and suffix.isdigit() else filename
        label = label.lower()
        labeled_files.append((file_path, label))

    return labeled_files


def prepare_data(data_dir: Path = DATA_DIR) -> list[list[str | float]]:
    """Return normalized, labeled landmark rows ready for CSV output."""
    prepared_rows = []

    for file_path, label in load_labeled_files(data_dir):
        for line in file_path.read_text(encoding="utf-8").splitlines():
            hand_match = HAND_LINE_PATTERN.search(line)
            if hand_match is None:
                continue

            landmarks = [
                tuple(float(value) for value in match)
                for match in LANDMARK_PATTERN.findall(hand_match.group(2))
            ]
            if len(landmarks) != 21:
                continue

            wrist = landmarks[0]
            scale = math.dist(wrist, landmarks[9])
            if scale == 0:
                continue

            row: list[str | float] = [label, hand_match.group(1)]
            for landmark in landmarks[1:]:
                row.extend(
                    (coordinate - wrist_coordinate) / scale
                    for coordinate, wrist_coordinate in zip(landmark, wrist)
                )
            prepared_rows.append(row)

    return prepared_rows


def write_training_data(
    output_path: Path = DATA_DIR / "hare_hunter.csv",
    data_dir: Path = DATA_DIR,
) -> Path:
    """Write normalized landmark rows and their header to a CSV file."""
    header = ["label", "handedness"]
    for landmark_index in range(1, 21):
        header.extend(
            f"{coordinate}_{landmark_index}"
            for coordinate in ("x", "y", "z")
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(prepare_data(data_dir))

    return output_path


def main() -> None:
    """Prepare collected hand landmark data for model training."""
    output_path = write_training_data()
    print(f"Training data written to {output_path}")


if __name__ == "__main__":
    main()
