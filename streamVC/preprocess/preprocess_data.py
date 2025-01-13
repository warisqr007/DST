import numpy as np
import argparse
from tqdm import tqdm
from functools import partial
from argparse import RawTextHelpFormatter
from pathlib import Path

from src.utils import attr_dict
from hubert.hubert_feature_from_audio import HubertFeatureReader, ApplyKmeans
from preprocess.energy_calculation import calculate_energy
from preprocess.pitch_calculation import calculate_pitch
import speaker.encoder as spkr_encoder



def preprocess_wavs(wav_files, h, device, exist_ok=False):
    _hubert = HubertFeatureReader(h.hubert_ckpt, h.hubert_layer)
    _kmeans = ApplyKmeans(h.km_path)

    spkr_encoder.load_model(device=device)

    for wav in tqdm(wav_files):
        # Calculate energy and pitch
        calculate_energy(wav, h, exist_ok)
        calculate_pitch(wav, h, exist_ok)

        # Calculate speaker embedding
        spkr_emb = spkr_encoder.generate_and_save_speaker_embedding(wav, exist_ok=exist_ok)
        
        # Calculate hubert features
        (Path(wav).parent.parent / 'hubert').mkdir(parents=True, exist_ok=True)
        hubert_path = (Path(wav).parent.parent / 'hubert'/ f"{Path(wav).stem}.hub.npy")
        if not exist_ok or not hubert_path.exists():
            feat = _hubert.get_feats(wav)
            np.save(hubert_path, feat.squeeze().cpu().numpy())

            lab = _kmeans(feat.cpu().numpy()).tolist()
            basename = Path(wav).stem
            lab_path = (Path(wav).parent.parent / 'code')
            lab_path.mkdir(parents=True, exist_ok=True)
            lab_path = lab_path / f"{basename}.km"
            with open(lab_path, "w") as f:
                f.write(" ".join(map(str, lab)))


def get_wave_files(dataset_path):
    wav_files = list(Path(dataset_path).rglob("*.wav"))
    return wav_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Compute embedding vectors for each wav file in a dataset.""",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("dataset_path", type=str, help="Path to dataset waves.")
    parser.add_argument("-c", "--config", help="config file", dest="config", default='experiments/base/config.json')
    parser.add_argument("-d", "--device", help="device for sprk encoder/hubert", dest="device", default='cuda')
    parser.add_argument("-e", "--exist_ok", help="overwrite existing files", dest="exist_ok", action='store_true', default=False)

    
    args = parser.parse_args()
    dataset_path = args.dataset_path

    wav_files = get_wave_files(dataset_path)
    h = attr_dict(args.config)
    
    preprocess_wavs(wav_files, h, args.device, args.exist_ok)


