import torch
import librosa
from scipy.io import wavfile
from speechbrain.inference import EncoderClassifier
import numpy as np
import json
import time
from pathlib import Path

from anonymization.anonymizer import DemoAnonymizer
from src.encoder import Encoder
from src.quantizer.bnftocode import Quantizer
from src.code_generator import CodeGenerator
from src.utils import AttrDict, load_checkpoint

MAX_WAV_VALUE = 32768.0


class ModelManager:
    def __init__(self, config_file, ckpt_file, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.config = self._load_config(config_file)
        self.encoder, self.generator, self.quantizer = self._load_models(ckpt_file)
        self.tdnn_model = self._load_speaker_model('speechbrain/spkrec-ecapa-voxceleb')
        self.xvec_model = self._load_speaker_model('speechbrain/spkrec-xvect-voxceleb')
        self.anonymizer = DemoAnonymizer(model_path=Path('anonymization/models'), model_tag='gan', device=self.device)
        print("All models loaded successfully.")

    @staticmethod
    def _load_config(config_file):
        with open(config_file, 'r') as f:
            return AttrDict(json.load(f))

    def _load_models(self, ckpt_file):
        encoder = Encoder(self.config).to(self.device)
        generator = CodeGenerator(self.config).to(self.device)

        state_dict_g = load_checkpoint(ckpt_file, self.device)
        encoder.load_state_dict(state_dict_g['encoder'])
        generator.load_state_dict(state_dict_g['generator'])
        encoder.eval().remove_weight_norm()
        generator.eval().remove_weight_norm()

        quantizer = Quantizer(hdim=self.config.upsample_initial_channel, codebook_size=self.config.vq_size).to(self.device)
        quantizer.load_state_dict(state_dict_g['quantizer'])
        quantizer.eval()
        print("Encoder, Generator, and Quantizer loaded.")

        return encoder, generator, quantizer

    def _load_speaker_model(self, model_path):
        model = EncoderClassifier.from_hparams(
            source=model_path,
            savedir=f'pretrained_models/speaker/{model_path.split("/")[-1]}',
            run_opts={'device': self.device}
        )
        model.eval()
        print(f"Speaker model {model_path} loaded.")
        return model


class AudioProcessor:
    def __init__(self, model_manager, sample_rate=16000):
        self.models = model_manager
        self.sample_rate = sample_rate

    def _load_audio(self, wav_file):
        audio, _ = librosa.load(wav_file, sr=self.sample_rate)
        return torch.from_numpy(audio).to(self.models.device)

    @torch.no_grad()
    def get_speaker_embed(self, wav_file):
        wave = self._load_audio(wav_file, 16000)
        spk_emb_x = self.models.xvec_model.encode_batch(wavs=wave.unsqueeze(0)).squeeze()
        spk_emb_t = self.models.tdnn_model.encode_batch(wavs=wave.unsqueeze(0)).squeeze()
        return torch.cat([spk_emb_x, spk_emb_t], dim=0).cpu().numpy()

    @torch.no_grad()
    def get_anon_embed(self, wav_file):
        wave = self._load_audio(wav_file, 16000)
        anon_spkr = self.models.anonymizer.anonymize_embedding(wave, 16000)
        anon_spkr = torch.cat([anon_spkr[192:], anon_spkr[:192]], dim=0)
        return anon_spkr.cpu().numpy()

    @torch.no_grad()
    def process_wav(self, wav_file, ref_wav_file=None, out_file=None):
        audio = self._load_audio(wav_file).unsqueeze(0).unsqueeze(0)
        embed = (self.get_anon_embed(wav_file) if ref_wav_file is None
                 else self.get_speaker_embed(ref_wav_file))
        spkr = torch.from_numpy(embed).to(self.models.device).unsqueeze(0)

        encoder_buf = self.models.encoder.init_buffers(1, self.models.device)
        x_hat, encoder_buf = self.models.encoder(audio, encoder_buf)
        x_hat_quantized, _, _ = self.models.quantizer.vq(x_hat.transpose(1, 2))

        feat = {'x_hat': x_hat_quantized.transpose(1, 2), 'spkr': spkr, 'pitch': None, 'energy': None}
        generator_buf = self.models.generator.init_buffers(1, self.models.device)
        wav, generator_buf = self.models.generator(generator_buf, is_inference=True, **feat)

        out_audio = (wav.squeeze() * MAX_WAV_VALUE).cpu().numpy().astype('int16')
        if out_file:
            wavfile.write(out_file, sample_rate, out_audio)
        else:
            return out_audio

    @torch.no_grad()
    def generate_wav2tokens(self, wav_file, out_file=None):
        audio = self._load_audio(wav_file).unsqueeze(0).unsqueeze(0)
        encoder_buf = self.models.encoder.init_buffers(1, self.models.device)
        x_hat, encoder_buf = self.models.encoder(audio, encoder_buf)
        x_hat_quantized, x_hat_indices, _ = self.models.quantizer.vq(x_hat.transpose(1, 2))

        if out_file:
            np.save(out_file, x_hat_indices.squeeze().cpu().numpy())
        else:
            return x_hat_indices.squeeze().cpu().numpy()


# Usage Example
if __name__ == "__main__":
    model_manager = ModelManager(config_file="config.json", ckpt_file="model.ckpt", qckpt_file="quant.ckpt")
    processor = AudioProcessor(model_manager)

    speaker_embed = processor.get_speaker_embed("input.wav")
    anon_embed = processor.get_anon_embed("input.wav")
    processor.process_wav("input.wav", out_file="output.wav")
    output_tokens = processor.generate_wav2tokens("input.wav")
    print("Output tokens:", output_tokens)
    print("All processing done.")
