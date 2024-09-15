```markdown
# Manacus Audio Processing

This repository contains a script for processing MP4 video files to extract audio clips, generate spectrograms, and classify them using a trained Keras model.

## Project Structure

```plaintext
manacusaudio/
├── audio_processing/
│   ├── __init__.py
│   ├── spectrogram.py
│   └── extract_clips.py
├── models/
│   └── audiomodel_88_85.keras
├── scripts/
│   └── predict.py
├── .gitignore
├── README.md
├── requirements.txt
```

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/marcocarbullido/manacusaudio.git
    cd manacusaudio
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Find a Keras model in the `models` directory named `audiomodel_88_85.keras`.

2. Run the prediction script with the paths to the model path and your video file:
    ```sh
    python scripts/predict.py models/audiomodel_88_85.keras path/to/your/video.mp4
    ```

## Notes

- The script will extract audio clips from the video, generate spectrograms for each clip, and classify them using the provided Keras model.
- Results will be saved in a `predictions.csv` file.
