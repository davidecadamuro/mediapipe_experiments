<video controls width="720">
	<source src="demo_good.mov" type="video/quicktime">
	Your browser does not support the video tag.
</video>

# Hare Hunter

Hare Hunter is a small learning project for exploring how MediaPipe detects hand landmarks and how those landmarks can be used with machine learning for gesture recognition.

The game recognizes three classes:

- `hare`
- `hunter`
- `garbage` for other hand gestures

## Setup

Create and activate the virtual environment, then install the dependencies:

```bash
source mp_exp.venv/bin/activate
python -m pip install -r requirements.txt
```

The game uses the Press Start 2P font from `fonts/PressStart2P-Regular.ttf`.

## Build The Model

### 1. Collect Samples

Run `collect_hand_data.py` once for each class. Hold the requested gesture in front of the camera and press the spacebar to save a sample. Press Escape to stop collecting.

Collect samples for `hare`, `hunter`, and random gestures for `garbage`:

```bash
python collect_hand_data.py -f hare.log
python collect_hand_data.py -f hunter.log
python collect_hand_data.py -f garbage.log
```

The log files are saved in `data/`. The filename is used as the class label. Existing filenames are protected from being overwritten; use a suffix for additional recordings, such as `hare_1.log` or `hare_2.log`.

### 2. Prepare Training Data

Parse and normalize the landmark logs, then write the training CSV:

```bash
python prepare_data.py
```

This creates:

```text
data/hare_hunter.csv
```

The prepared data is centered on the wrist landmark and scale-normalized before training.

### 3. Train The Model

Train the Random Forest model from the prepared CSV:

```bash
python train_model.py -f hare_hunter.csv
```

This creates:

```text
models/hare_hunter.cpickle
```

## Run The Game

Once the model has been trained, start the game:

```bash
python hare_hunter_game.py
```

Show one hand classified as `hare` and the other as `hunter` to start a 30-second round. The game stores qualifying scores in `leaderboard.csv`.
