from sklearn.preprocessing import StandardScaler
import glob
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
import os
from tqdm import tqdm
import json


def build_from_path(in_dir):
    print("Processing Data ...")
    pitch_scaler = StandardScaler()

    files = glob.glob(os.path.join(in_dir, "**/*.pit.npy"), recursive=True)

    for i, file in enumerate(tqdm(files)):
        try:
            pitch = np.load(file)
        except:
            print(f"Error in file {file}")
            continue
        if len(pitch) > 0:
            pitch_scaler.partial_fit(pitch.reshape((-1, 1)))

    print("Computing statistic quantities ...")
    # Perform normalization if necessary
    pitch_mean = pitch_scaler.mean_[0]
    pitch_std = pitch_scaler.scale_[0]

    pitch_min, pitch_max = normalize_pitch(files, pitch_mean, pitch_std)

    with open(os.path.join(in_dir, "metadata/stats.json"), "w") as f:
        stats = {
            "pitch": [
                float(pitch_min),
                float(pitch_max),
                float(pitch_mean),
                float(pitch_std),
            ]
        }
        f.write(json.dumps(stats))


def normalize_pitch(files, mean, std):
    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max

    print("Normalizing pitch...")
    for file in tqdm(files):
        values = (np.load(file) - mean) / std
        np.save(file, values)

        max_value = max(max_value, max(values))
        min_value = min(min_value, min(values))

    return min_value, max_value

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Normalize pitch values.""",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("dataset_path", type=str, help="Path to dataset directory.")
    
    args = parser.parse_args()
    dataset_path = args.dataset_path
    
    build_from_path(dataset_path)