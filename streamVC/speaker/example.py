import torch
import librosa
from speechbrain.inference import EncoderClassifier


class SpeechBrainEncoder:
    def __init__(self, device=None, sample_rate=16000):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tdnn_model = self._load_speaker_model('speechbrain/spkrec-ecapa-voxceleb')
        self.xvec_model = self._load_speaker_model('speechbrain/spkrec-xvect-voxceleb')
        self.sample_rate = sample_rate

    def _load_speaker_model(self, model_path):
        model = EncoderClassifier.from_hparams(
            source=model_path,
            savedir=f'pretrained_models/speaker/{model_path.split("/")[-1]}',
            run_opts={'device': self.device}
        )
        model.eval()
        print(f"Speaker model {model_path} loaded.")
        return model

    def _load_audio(self, wav_file):
        audio, _ = librosa.load(wav_file, sr=self.sample_rate)
        return torch.from_numpy(audio).to(self.models.device)

    @torch.no_grad()
    def get_speaker_embed(self, wav_file):
        wave = self._load_audio(wav_file)
        spk_emb_x = self.models.xvec_model.encode_batch(wavs=wave.unsqueeze(0)).squeeze()
        spk_emb_t = self.models.tdnn_model.encode_batch(wavs=wave.unsqueeze(0)).squeeze()
        return torch.cat([spk_emb_x, spk_emb_t], dim=0).cpu().numpy()


if __name__ == '__main__':
    encoder = SpeechBrainEncoder()
    spk_embed = encoder.get_speaker_embed('path/to/wav/file')
    print(spk_embed.shape)