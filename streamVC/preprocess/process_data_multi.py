import numpy as np
import argparse
from tqdm import tqdm
from functools import partial
from argparse import RawTextHelpFormatter
from pathlib import Path
from multiprocessing import Pool, cpu_count, current_process
from src.utils import attr_dict
from hubert.hubert_feature_from_audio import HubertFeatureReader, ApplyKmeans
from preprocess.energy_calculation import calculate_energy
from preprocess.pitch_calculation import calculate_pitch
import speaker.encoder as spkr_encoder

# Globals to store models for each worker
_hubert = None
_kmeans = None
_spkr_encoder = None

def init_worker(h, device):
    """Initialize models once per worker process."""
    global _hubert, _kmeans, _spkr_encoder
    _hubert = HubertFeatureReader(h.hubert_ckpt, h.hubert_layer)
    _kmeans = ApplyKmeans(h.km_path)
    spkr_encoder.load_model(device=device)
    _spkr_encoder = spkr_encoder


def process_single_wav(wav, h):
    """Process a single WAV file."""
    global _hubert, _kmeans, _spkr_encoder

    # Calculate energy and pitch
    calculate_energy(wav, h)
    calculate_pitch(wav, h)

    # Calculate speaker embedding
    spkr_emb = _spkr_encoder.generate_and_save_speaker_embedding(wav)
    
    # Calculate hubert features
    feat = _hubert.get_feats(wav)
    hubert_path = (Path(wav).parent.parent / 'hubert')
    hubert_path.mkdir(parents=True, exist_ok=True)
    np.save(hubert_path / f"{Path(wav).stem}.hub.npy", feat.squeeze().cpu().numpy())

    lab = _kmeans(feat.cpu().numpy()).tolist()
    basename = Path(wav).stem
    lab_path = (Path(wav).parent.parent / 'code')
    lab_path.mkdir(parents=True, exist_ok=True)
    lab_path = lab_path / f"{basename}.km"
    with open(lab_path, "w") as f:
        f.write(" ".join(map(str, lab)))


def preprocess_wavs(wav_files, h, device, num_workers):
    # Initialize partial function for single WAV processing
    process_fn = partial(process_single_wav, h=h)

    # Use multiprocessing Pool and initialize models once per worker
    with Pool(processes=num_workers, initializer=init_worker, initargs=(h, device)) as pool:
        list(tqdm(pool.imap_unordered(process_fn, wav_files), total=len(wav_files)))


def get_wave_files(dataset_path):
    """Retrieve all WAV files from the dataset path."""
    return list(Path(dataset_path).rglob("*.wav"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Compute embedding vectors for each wav file in a dataset.""",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("dataset_path", type=str, help="Path to dataset waves.")
    parser.add_argument("-c", "--config", help="config file", dest="config", default='experiments/base/config.json')
    parser.add_argument("-d", "--device", help="device for sprk encoder/hubert", dest="device", default='cuda')
    parser.add_argument("-n", "--num_workers", help="Number of worker processes", type=int, default=cpu_count())

    args = parser.parse_args()
    dataset_path = args.dataset_path

    wav_files = get_wave_files(dataset_path)
    h = attr_dict(args.config)
    
    preprocess_wavs(wav_files, h, args.device, args.num_workers)
