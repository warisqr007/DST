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

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def mel_spectogram_sb(audio, sample_rate, hop_length, win_length, n_fft, n_mels,
                      f_min, f_max, power=1, normalized=False, min_max_energy_norm=True,
                      norm="slaney", mel_scale="slaney", compression=True,
):
    """calculates MelSpectrogram for a raw audio signal
    Arguments
    ---------
    sample_rate : int
        Sample rate of audio signal.
    hop_length : int
        Length of hop between STFT windows.
    win_length : int
        Window size.
    n_fft : int
        Size of FFT.
    n_mels : int
        Number of mel filterbanks.
    f_min : float
        Minimum frequency.
    f_max : float
        Maximum frequency.
    power : float
        Exponent for the magnitude spectrogram.
    normalized : bool
        Whether to normalize by magnitude after stft.
    norm : str or None
        If "slaney", divide the triangular mel weights by the width of the mel band
    mel_scale : str
        Scale to use: "htk" or "slaney".
    compression : bool
        whether to do dynamic range compression
    audio : torch.tensor
        input audio signal
    """
    from torchaudio import transforms

    audio_to_mel = transforms.Spectrogram(
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        power=power,
        normalized=normalized,
    ).to(audio.device)

    mel_scale = transforms.MelScale(
        sample_rate=sample_rate,
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        norm=norm,
        mel_scale=mel_scale,
    ).to(audio.device)
    spec = audio_to_mel(audio)
    mel = mel_scale(spec)
    assert mel.dim() == 2
    assert mel.shape[0] == n_mels
    rmse = torch.norm(mel, dim=0)

    if min_max_energy_norm:
        rmse = (rmse - torch.min(rmse)) / (torch.max(rmse) - torch.min(rmse))

    if compression:
        mel = dynamic_range_compression_torch(mel)

    return mel, rmse

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def get_dataset_filelist(h):
    training_files = list(Path(h.training_fpath).rglob("*.wav"))
    training_files, validation_files, _, _ = train_test_split(training_files, training_files, test_size=0.01, random_state=42)
    # validation_files = list(Path(h.training_fpath).rglob("*.wav"))

    print("Training Files: ", len(training_files))
    print("Validation Files: ", len(validation_files))
    return training_files, validation_files,

def get_dataset_filelist_vq(h):
    training_files = list(Path(h.training_fpath).rglob("*.hub.npy"))
    training_files, validation_files, _, _ = train_test_split(training_files, training_files, test_size=0.01, random_state=42)

    print("Training Files: ", len(training_files))
    print("Validation Files: ", len(validation_files))
    return training_files, validation_files,


class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, code_hop_size, 
                 sampling_rate, split=True, device=None):

        self.audio_files = training_files
        random.seed(1234)
        self.segment_size = segment_size
        self.code_hop_size = code_hop_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.device = device

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
        tokens = wav_fpath.parent.parent / "code" / f"{wav_fpath.stem}.km"

        tokens = open(tokens, 'r').readlines()[0].strip()
        tokens = np.array([int(i) for i in tokens.split(" ")])

        audio, sampling_rate = load_audio(wav_fpath)
        if sampling_rate != self.sampling_rate:
            import resampy
            audio = resampy.resample(audio, sampling_rate, self.sampling_rate)

        # audio = audio / MAX_WAV_VALUE
        # audio = normalize(audio) * 0.95
        audio = audio / (max(abs(audio)) + 0.001) * 0.95

        # Trim audio ending
        code_length = min(audio.shape[0] // self.code_hop_size, tokens.shape[-1])
        tokens = tokens[:code_length]
        
        audio = audio[:code_length * self.code_hop_size]
        
        assert audio.shape[0] // self.code_hop_size == tokens.shape[-1], "Code audio mismatch"

        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            tokens = np.hstack([tokens, tokens])

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        tokens = torch.from_numpy(tokens)

        assert audio.size(1) >= self.segment_size, "Padding not supported!!"
        
        audio, tokens = self._sample_interval([audio, tokens])

        return audio.squeeze(0), tokens, str(wav_fpath)

    def __len__(self):
        return len(self.audio_files)


class VQCodeDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, code_hop_size, 
                 sampling_rate, split=True, device=None):

        self.audio_files = training_files
        random.seed(1234)
        self.segment_size = segment_size
        self.code_hop_size = code_hop_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.device = device

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
        base_path = wav_fpath.stem.split(".")[0]
        wav_fpath = wav_fpath.parent.parent / "wav" / f"{base_path}.wav"
        tokens = wav_fpath.parent.parent / "code" / f"{base_path}.km"
        hubert_feats = wav_fpath.parent.parent / "hubert" / f"{base_path}.hub.npy"

        tokens = open(tokens, 'r').readlines()[0].strip()
        tokens = np.array([int(i) for i in tokens.split(" ")])

        audio, sampling_rate = load_audio(wav_fpath)
        if sampling_rate != self.sampling_rate:
            import resampy
            audio = resampy.resample(audio, sampling_rate, self.sampling_rate)

        # audio = audio / MAX_WAV_VALUE
        # audio = normalize(audio) * 0.95
        audio = audio / (max(abs(audio)) + 0.001) * 0.95
        hubert_feats = np.load(hubert_feats).T

        # Trim audio ending
        code_length = min(audio.shape[0] // self.code_hop_size, tokens.shape[-1])
        tokens = tokens[:code_length]
        hubert_feats = hubert_feats[:, :code_length]
        
        audio = audio[:code_length * self.code_hop_size]
        
        assert audio.shape[0] // self.code_hop_size == tokens.shape[-1], "Code audio mismatch"

        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            tokens = np.hstack([tokens, tokens])
            hubert_feats = np.hstack([hubert_feats, hubert_feats])

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        tokens = torch.from_numpy(tokens)

        assert audio.size(1) >= self.segment_size, "Padding not supported!!"
        
        audio, hubert_feats = self._sample_interval([audio, hubert_feats])

        return audio.squeeze(0), hubert_feats, str(wav_fpath)

    def __len__(self):
        return len(self.audio_files)



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


class VQCodeDatasetPED(torch.utils.data.Dataset):
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
        hubert_feats = wav_fpath.parent.parent / "hubert" / f"{wav_fpath.stem}.hub.npy"

        tokens = open(tokens, 'r').readlines()[0].strip()
        tokens = np.array([int(i) for i in tokens.split(" ")])
        code = self.C_np[tokens].transpose()
            
        spkr_embed = np.load(spkr_embed)
        spkr_embed = torch.FloatTensor(spkr_embed)

        hubert_feats = np.load(hubert_feats).T

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
        hubert_feats = hubert_feats[:, :code_length]
        
        audio = audio[:code_length * self.code_hop_size]
        
        assert audio.shape[0] // self.code_hop_size == code.shape[-1], "Code audio mismatch"

        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            code = np.hstack([code, code])
            pitch = np.hstack([pitch, pitch])
            energy = np.hstack([energy, energy])
            tokens = np.hstack([tokens, tokens])
            hubert_feats = np.hstack([hubert_feats, hubert_feats])

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        pitch = torch.FloatTensor(pitch)
        energy = torch.FloatTensor(energy)
        tokens = torch.from_numpy(tokens)

        assert audio.size(1) >= self.segment_size, "Padding not supported!!"
        
        audio, code, pitch, energy, tokens, hubert_feats = self._sample_interval([audio, code, pitch, energy, tokens, hubert_feats])

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        feats = {"code": code.squeeze(),
                 "spkr": spkr_embed,
                 "pitch": pitch,
                 "energy": energy,
                 "tokens": tokens,
                 "hubert": hubert_feats}

        return feats, audio.squeeze(0), str(wav_fpath), mel_loss.squeeze()

    def __len__(self):
        return len(self.audio_files)