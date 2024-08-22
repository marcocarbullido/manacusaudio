import librosa
import numpy as np

def spect(wav_filepath):
    y, sr = librosa.load(wav_filepath, mono=True)
    y_res = 100
    D = librosa.stft(y, n_fft=y_res*2-1, hop_length=32)
    magnitude = np.abs(D)
    log_magnitude = librosa.amplitude_to_db(magnitude)
    log_magnitude_normalized = (log_magnitude - np.mean(log_magnitude)) / np.std(log_magnitude)
    return log_magnitude_normalized
