import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"


def get_training_paths() -> tuple[Path, Path]:
    """Return the selected CSV path and its corresponding model path."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="CSV filename from the data directory.",
    )
    args = parser.parse_args()

    filename = Path(args.filename)
    if filename.name != args.filename or filename.suffix.lower() != ".csv":
        parser.error("filename must be a CSV filename, for example hare_hunter.csv")

    data_path = DATA_DIR / filename.name
    if not data_path.is_file():
        parser.error(f"data file does not exist: {data_path}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{filename.stem}.cpickle"
    return data_path, model_path


def load_training_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load labels and numeric features from the prepared CSV file."""
    features = []
    labels = []

    with data_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        feature_columns = reader.fieldnames[2:]
        for row in reader:
            handedness = 1.0 if row["handedness"] == "Right" else 0.0
            coordinates = [float(row[column]) for column in feature_columns]
            features.append([handedness, *coordinates])
            labels.append(row["label"])

    return np.asarray(features, dtype=float), np.asarray(labels)


def split_training_data(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into stratified training, validation, and test sets."""
    train_features, temporary_features, train_labels, temporary_labels = (
        train_test_split(
            features,
            labels,
            test_size=0.3,
            random_state=random_state,
            stratify=labels,
        )
    )
    validation_features, test_features, validation_labels, test_labels = (
        train_test_split(
            temporary_features,
            temporary_labels,
            test_size=0.5,
            random_state=random_state,
            stratify=temporary_labels,
        )
    )
    return (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        test_features,
        test_labels,
    )


def train_random_forest(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Fit a random forest classifier on the training data."""
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_features, train_labels)
    return model


def save_model(model: RandomForestClassifier, model_path: Path) -> None:
    """Serialize a trained model to disk."""
    with model_path.open("wb") as model_file:
        pickle.dump(model, model_file)


def main() -> None:
    """Train a gesture-recognition model."""
    data_path, model_path = get_training_paths()
    features, labels = load_training_data(data_path)
    datasets = split_training_data(features, labels)
    train_features, train_labels = datasets[0], datasets[1]
    validation_features, validation_labels = datasets[2], datasets[3]
    test_features, test_labels = datasets[4], datasets[5]

    print(f"Training data: {data_path}")
    print(f"Model output: {model_path}")
    print(f"Training set: {train_features.shape}, {train_labels.shape}")
    print(f"Validation set: {validation_features.shape}, {validation_labels.shape}")
    print(f"Test set: {test_features.shape}, {test_labels.shape}")

    model = train_random_forest(train_features, train_labels)
    save_model(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
