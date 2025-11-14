import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from audio_processing.spectrogram import spect

LABEL_MAP = {0: 'c', 1: 'w', 2: 'g', 3: 'n', 4: 's'}


@dataclass
class AudioModelContext:
    model: tf.keras.Model
    label_map: Dict[int, str]
    target_time: int
    target_freq: int
    expects_channel_axis: bool


def _resolve_input_shape(model: tf.keras.Model) -> Tuple[int, int, bool]:
    """Return (time_steps, freq_bins, expects_channel_axis) from the loaded model."""
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        if not input_shape:
            raise ValueError("Model reports an empty input shape list.")
        input_shape = input_shape[0]

    if len(input_shape) == 3:
        _, time_steps, freq_bins = input_shape
        expects_channel_axis = False
    elif len(input_shape) == 4:
        _, time_steps, freq_bins, _ = input_shape
        expects_channel_axis = True
    else:
        raise ValueError(f"Unsupported input shape from model: {input_shape}")

    return time_steps, freq_bins, expects_channel_axis


def _prepare_spectrogram_for_model(raw_spectrogram, target_time, target_freq, expects_channel_axis):
    """Resize/pad the spectrogram so the model can ingest variable-duration clips."""
    spec = np.asarray(raw_spectrogram, dtype=np.float32)
    spec = np.transpose(spec, (1, 0))  # convert to (time, freq)

    original_time, original_freq = spec.shape
    resize_time = target_time if target_time is not None else original_time
    resize_freq = target_freq if target_freq is not None else original_freq

    tensor = tf.convert_to_tensor(spec)[tf.newaxis, ..., tf.newaxis]  # (1, time, freq, 1)
    tensor = tf.image.resize(tensor, [resize_time, resize_freq], method="bilinear")

    if expects_channel_axis:
        prepared = tensor.numpy()
    else:
        prepared = tf.squeeze(tensor, axis=-1).numpy()

    return prepared


def load_audio_model(model_path: str) -> AudioModelContext:
    """Load the audio model once so downstream callers can run many clips efficiently."""
    model = tf.keras.models.load_model(model_path)
    target_time, target_freq, expects_channel_axis = _resolve_input_shape(model)
    return AudioModelContext(
        model=model,
        label_map=LABEL_MAP,
        target_time=target_time,
        target_freq=target_freq,
        expects_channel_axis=expects_channel_axis,
    )


def predict_with_context(ctx: AudioModelContext, clip_path: str, return_probabilities: bool = False):
    """Predict using a pre-loaded AudioModelContext; optionally return raw probabilities."""
    raw_spectrogram = spect(clip_path)
    model_input = _prepare_spectrogram_for_model(
        raw_spectrogram,
        target_time=ctx.target_time,
        target_freq=ctx.target_freq,
        expects_channel_axis=ctx.expects_channel_axis,
    )

    predictions = ctx.model.predict(model_input, verbose=0)
    probabilities = predictions[0]
    max_prob_index = int(np.argmax(probabilities))
    label = ctx.label_map[max_prob_index]
    if return_probabilities:
        return label, probabilities
    return label


def predict_audio_clip(model_path, clip_path):
    """Load the trained model, resize any clip length to the expected shape, and return the top label."""
    ctx = load_audio_model(model_path)
    return predict_with_context(ctx, clip_path)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python predict.py <model_path> <audio_clip_path>")
    else:
        model_path = sys.argv[1]
        clip_path = sys.argv[2]
        prediction = predict_audio_clip(model_path, clip_path)
        print(prediction)
