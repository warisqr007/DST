import os
import torch
import torchaudio
import numpy as np
from pathlib import Path


def calculate_pitch(wav, h, exist_ok=False):
    audio, fs = torchaudio.load(wav)
    # Computes pitch
    (Path(wav).parent.parent / 'pitch').mkdir(parents=True, exist_ok=True)
    pitch_file = Path(wav).parent.parent / 'pitch' / f"{Path(wav).stem}.pit.npy"

    if exist_ok and os.path.isfile(pitch_file):
        return
    else:
        pitch = torchaudio.functional.detect_pitch_frequency(
            waveform=audio,
            sample_rate=fs,
            frame_time=(h.pitch_hop_length / fs),
            win_length=3,
            freq_low=h.pitch_min_f0,
            freq_high=h.pitch_max_f0,
        ).squeeze(0)

        # Concatenate last element to match duration.
        pitch = torch.cat([pitch, pitch[-1].unsqueeze(0)])

        # # Mean and Variance Normalization
        # mean = 256.1732939688805
        # std = 328.319759158607

        # pitch = (pitch - mean) / std
        np.save(pitch_file, pitch)