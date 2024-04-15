import wave
import numpy as np
from pathlib import Path
from typing import Tuple


def spectrogram(samples, frame_len=256, frame_step=128):
    """
    Computes a magnitude spectrogram for a given vector of samples at a given
    frame length (in samples) and frame step (in samples).
    """
    rfft = np.fft.rfft
    
    if len(samples) < frame_len:
        return np.empty((0, frame_len // 2 + 1), dtype=samples.dtype)

    win = np.hanning(frame_len).astype(samples.dtype)
    num_frames = max(0, (len(samples) - frame_len) // frame_step + 1)
    rfft_samples = [np.abs(rfft(samples[pos:pos + frame_len] * win))
                    for pos in range(0, len(samples) - frame_len + 1,
                                     int(frame_step))]
    spect = np.vstack(rfft_samples)

    return spect


# Utility function that reads the whole `wav` file content into a numpy array
def wave_read(filename: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(filename), 'rb') as f:
        buffer = f.readframes(f.getnframes())
        inter = np.frombuffer(buffer, dtype=f'int{f.getsampwidth()*8}')
        dtype=f'int{f.getsampwidth()*8}'
        print(dtype)
        return np.reshape(inter, (-1, f.getnchannels())), f.getframerate()


if __name__ == '__main__':
	DATASET_PATH = '../data'

	data_dir = Path(DATASET_PATH)

	wav, frame_rate = wave_read(data_dir/'test/h_yes.wav')
	wav = wav.flatten()              # 1-D array
	wav = wav / np.linalg.norm(wav)  # normalize

	np_spec = spectrogram(wav)
	print(np_spec.shape, np_spec.dtype, np_spec[0,:10])
