import os
import librosa
import torch
import numpy as np
from pathlib import Path

from src.dataset import mel_spectogram_sb


def calculate_energy(wav, h, exist_ok=False):
    # Computes energy
    (Path(wav).parent.parent / 'energy').mkdir(parents=True, exist_ok=True)
    energy_file = Path(wav).parent.parent / 'energy' / f"{Path(wav).stem}.eng.npy"

    if exist_ok and os.path.isfile(energy_file):
        return
    else:
        audio, _ = librosa.load(wav, sr = h.sampling_rate)
        audio = torch.from_numpy(audio)
        _, energy = mel_spectogram_sb(audio, h.sampling_rate, h.code_hop_size, h.win_size, 
                                      h.n_fft, h.num_mels, h.fmin, h.fmax)
        np.save(energy_file, energy)
