import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from audio_processing.spectrogram import spect
from audio_processing.extract_clips import extract_audio_clips


def predict_audio_clips(model_path, video_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    model.summary()

    # Extract audio clips
    clips_output_dir = os.path.join('output_clips', os.path.basename(video_path).split('.')[0])
    extract_audio_clips(video_path, clips_output_dir)

    audio_clips = [os.path.join(clips_output_dir, f) for f in os.listdir(clips_output_dir) if f.endswith('.wav')]
    index_to_label_map = {0: 'c', 1: 'w', 2: 'g', 3: 'n', 4: 's'}

    prediction_results = []

    for audio_clip in audio_clips:
        try:
            spectrogram = spect(audio_clip)
            spectrogram = np.array(spectrogram).reshape(1, *spectrogram.shape, 1)
            spectrogram = np.transpose(spectrogram, (0, 2, 1, 3))
            spectrogram = np.squeeze(spectrogram, axis=-1)

            predictions = model.predict(spectrogram)
            predicted_probabilities = predictions[0]
            predicted_labels = [index_to_label_map[idx] for idx in range(len(predicted_probabilities))]
            max_prob_index = np.argmax(predicted_probabilities)
            max_prob_label = predicted_labels[max_prob_index]

            prediction_result = {
                'path': audio_clip,
                'max_label': max_prob_label
            }
            for idx, prob in enumerate(predicted_probabilities):
                prediction_result[index_to_label_map[idx]] = prob

            prediction_results.append(prediction_result)
            
        except Exception as e:
            print(f"Error processing {audio_clip}: {e}")
            continue

    results_df = pd.DataFrame(prediction_results)
    results_df.to_csv('predictions.csv', index=False)
    #shutil.rmtree(clips_output_dir)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python predict.py <model_path> <video_path>")
    else:
        model_path = sys.argv[1]
        video_path = sys.argv[2]
        predict_audio_clips(model_path, video_path)
