import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.utils.data
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
import librosa
import joblib
from sklearn.model_selection import train_test_split

MAX_WAV_VALUE = 32768.0

def load_audio(full_path):
    data, sampling_rate = librosa.load(full_path, sr = 16000)
    return data, sampling_rate


def get_dataset_filelist(h):
    training_files = list(Path(h.training_fpath).rglob("*.wav"))
    training_files, validation_files, _, _ = train_test_split(training_files, training_files, test_size=0.01, random_state=42)
    # validation_files = list(Path(h.training_fpath).rglob("*.wav"))

    print("Training Files: ", len(training_files))
    print("Validation Files: ", len(validation_files))
    return training_files, validation_files,

    
class CodeDatasetPED(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, code_hop_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, km_path, split=True,
                 device=None, fmax_loss=None):
        self.audio_files = training_files
        random.seed(1234)
        self.segment_size = segment_size
        self.code_hop_size = code_hop_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.device = device
        self.C_np = self._load_cluster_centres(km_path)

    def _load_cluster_centres(self, km_path):
        km_model = joblib.load(km_path)
        C_np = km_model.cluster_centers_ #.transpose()
        return C_np

    def _sample_interval(self, seqs, seq_len=None):
        N = max([v.shape[-1] for v in seqs])
        if seq_len is None:
            seq_len = self.segment_size if self.segment_size > 0 else N

        hops = [N // v.shape[-1] for v in seqs]
        lcm = np.lcm.reduce(hops)

        # Randomly pickup with the batch_max_steps length of the part
        interval_start = 0
        interval_end = N // lcm - seq_len // lcm

        start_step = random.randint(interval_start, interval_end)

        new_seqs = []
        for i, v in enumerate(seqs):
            start = start_step * (lcm // hops[i])
            end = (start_step + seq_len // lcm) * (lcm // hops[i])
            new_seqs += [v[..., start:end]]

        return new_seqs

    def __getitem__(self, index):
        wav_fpath = self.audio_files[index]
        spkr_embed = wav_fpath.parent.parent / "spkr" / f"{wav_fpath.stem}.spk.npy"
        pitch = wav_fpath.parent.parent / "pitch" / f"{wav_fpath.stem}.pit.npy"
        energy = wav_fpath.parent.parent / "energy" / f"{wav_fpath.stem}.eng.npy"
        tokens = wav_fpath.parent.parent / "code" / f"{wav_fpath.stem}.km"

        tokens = open(tokens, 'r').readlines()[0].strip()
        tokens = np.array([int(i) for i in tokens.split(" ")])
        code = self.C_np[tokens].transpose()
            
        spkr_embed = np.load(spkr_embed)
        spkr_embed = torch.FloatTensor(spkr_embed)

        pitch = np.load(pitch)
        energy = np.load(energy)

        audio, sampling_rate = load_audio(wav_fpath)
        if sampling_rate != self.sampling_rate:
            import resampy
            audio = resampy.resample(audio, sampling_rate, self.sampling_rate)

        # if self.pad:
        #     padding = self.pad - (audio.shape[-1] % self.pad)
        #     audio = np.pad(audio, (0, padding), "constant", constant_values=0)
        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95

        # Trim audio ending
        code_length = min(audio.shape[0] // self.code_hop_size, code.shape[-1])
        code = code[:, :code_length]
        pitch = pitch[:code_length]
        energy = energy[:code_length]
        tokens = tokens[:code_length]
        
        audio = audio[:code_length * self.code_hop_size]
        
        assert audio.shape[0] // self.code_hop_size == code.shape[-1], "Code audio mismatch"

        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            code = np.hstack([code, code])
            pitch = np.hstack([pitch, pitch])
            energy = np.hstack([energy, energy])
            tokens = np.hstack([tokens, tokens])

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        pitch = torch.FloatTensor(pitch)
        energy = torch.FloatTensor(energy)
        tokens = torch.from_numpy(tokens)

        assert audio.size(1) >= self.segment_size, "Padding not supported!!"
        
        audio, code, pitch, energy, tokens = self._sample_interval([audio, code, pitch, energy, tokens])

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        feats = {"code": code.squeeze(),
                 "spkr": spkr_embed,
                 "pitch": pitch,
                 "energy": energy,
                 "tokens": tokens}

        return feats, audio.squeeze(0), str(wav_fpath), mel_loss.squeeze()

    def __len__(self):
        return len(self.audio_files)