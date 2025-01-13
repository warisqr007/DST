import logging
import os
import sys
import joblib
import glob
from pathlib import Path

from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.data.audio.audio_utils import get_features_or_waveform
from scipy.io.wavfile import read
import resampy

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")


class HubertFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000):
        (
            model,
            cfg,
            task,
        ) = load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        sampling_rate, wav = read(path)
        if self.task.cfg.sample_rate!=sampling_rate:
            wav = resampy.resample(wav, sampling_rate, self.task.cfg.sample_rate)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len=ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)

class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


# def main(in_dir, ckpt_path, layer, km_path, max_chunk):
#     reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
#     apply_kmeans = ApplyKmeans(km_path)

#     speakers = os.listdir(in_dir)
#     for speaker in tqdm(speakers):
#         files = glob.glob(f"{in_dir}/{speaker}/wav/*.wav")
#         os.makedirs(os.path.join(in_dir, speaker, 'text', 'l1_text'), exist_ok=True)
#         for file in files:
#             feat = reader.get_feats(file)
#             lab = apply_kmeans(feat.cpu().numpy()).tolist()
#             basename = Path(file).stem
#             lab_path = os.path.join(in_dir, speaker, 'text', 'l1_text', f"{basename}.km")
#             with open(lab_path, "w") as f:
#                 f.write(" ".join(map(str, lab)))
        
#         files = glob.glob(f"{in_dir}/{speaker}/o_wav/*.wav")
#         os.makedirs(os.path.join(in_dir, speaker, 'text', 'l2_text'), exist_ok=True)
#         for file in files:
#             feat = reader.get_feats(file)
#             lab = apply_kmeans(feat.cpu().numpy()).tolist()
#             basename = Path(file).stem
#             lab_path = os.path.join(in_dir, speaker, 'text', 'l2_text', f"{basename}.km")
#             with open(lab_path, "w") as f:
#                 # f.write(" ".join(map(str, lab)))
