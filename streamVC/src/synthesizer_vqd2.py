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
from src.quantizer.fsq import DownsampleFiniteScalarQuantize
from src.code_generator2 import CodeGenerator
from src.utils import AttrDict, load_checkpoint


MAX_WAV_VALUE = 32768.0
_encoder = None # type: Encoder
_quantizer = None # type: DownsampleFiniteScalarQuantize
_generator = None # type: CodeGenerator
_xvec_model = None # type: EncoderClassifier
_tdnn_model = None # type: EncoderClassifier
_anonymizer = None # type: DemoAnonymizer
_device = None # type: torch.device

def attr_dict(config_file):
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    return h


def load_models(config, ckpt, device=None):
    """
    Loads the model in memory. If this function is not explicitely called, it will be run on the 
    first call to embed_frames() with the default weights file.
    
    :param weights_fpath: the path to saved model weights.
    :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda"). The 
    model will be loaded and will run on this device. Outputs will however always be on the cpu. 
    If None, will default to your GPU if it"s available, otherwise your CPU.
    """
    # TODO: I think the slow loading of the encoder might have something to do with the device it
    #   was saved on. Worth investigating.
    global _encoder, _quantizer, _generator, _xvec_model, _tdnn_model, _anonymizer, _device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _device = torch.device(device)
    elif isinstance(device, str):
        _device = torch.device(device)

    _h = attr_dict(config)
    _encoder = Encoder(_h).to(device)
    _quantizer = DownsampleFiniteScalarQuantize(_h.upsample_initial_channel, _h.fsq_levels).to(device)
    _generator = CodeGenerator(_h).to(_device)

    state_dict_g = load_checkpoint(ckpt, _device)
    _encoder.load_state_dict(state_dict_g['encoder'])
    _quantizer.load_state_dict(state_dict_g['quantizer'])
    _generator.load_state_dict(state_dict_g['generator'])
    _encoder.eval()
    _generator.eval()
    _encoder.remove_weight_norm()
    _generator.remove_weight_norm()
    print("Loaded Encoder and Generator models.")

    _tdnn_model = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',
                                                 savedir='pretrained_models/speaker/spkrec-ecapa-voxceleb',
                                                 run_opts={'device': device})
    _xvec_model = EncoderClassifier.from_hparams(source='speechbrain/spkrec-xvect-voxceleb',
                                                 savedir='pretrained_models/speaker/spkrec-xvect-voxceleb',
                                                 run_opts={'device': device})
    _tdnn_model.eval()
    _xvec_model.eval()
    print("Loaded Speaker encoder.")

    _anonymizer = DemoAnonymizer(
        model_path=Path('anonymization/models'), 
        model_tag='gan',
        device=_device)
    print("Loaded Anonymizer.")
    
    
def is_loaded():
    return _encoder is not None and _generator is not None and _tdnn_model is not None and _xvec_model is not None and _anonymizer is not None


@torch.no_grad()
def get_speaker_embed(wav_file):
    wave, _ = librosa.load(wav_file, sr=16000)
    wave = torch.tensor(np.trim_zeros(wave))

    spk_emb_x = _xvec_model.encode_batch(wavs=wave.unsqueeze(0)).squeeze()
    spk_emb_t = _tdnn_model.encode_batch(wavs=wave.unsqueeze(0)).squeeze()
    embed = torch.cat([spk_emb_x, spk_emb_t], dim=0)
    embed = embed.cpu().numpy()
    return embed

@torch.no_grad()
def get_anon_embed(wav_file):
    wave, _ = librosa.load(wav_file, sr=16000)

    _anon_spkr = _anonymizer.anonymize_embedding(wave, 16000)
    anon_spkr = torch.cat([_anon_spkr[192:], _anon_spkr[:192]], dim=0)

    anon_spkr = anon_spkr.cpu().numpy()
    return anon_spkr

@torch.no_grad()
def process_wav(wav_file, ref_wav_file=None, out_file=None, sample_rate:int=16000):
    """
    Args:
        wav_file: Path to audio wav file
        sample_rate: target sample rate
    
    returns:
        2D BNF vector (H, T)
    """
    audio, _ = librosa.load(wav_file, sr=sample_rate)
    # audio = audio / MAX_WAV_VALUE

    audio = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(_device)

    #Spkr Embed
    if ref_wav_file is None:
        embed = get_anon_embed(wav_file)
    else:
        embed = get_speaker_embed(ref_wav_file)
    spkr = torch.from_numpy(embed).to(_device).unsqueeze(0)

    #Encoder
    encoder_buf = _encoder.init_buffers(1, _device)
    x_hat, encoder_buf = _encoder(audio, encoder_buf)

    # Quantizer
    x_hat_q, _ = _quantizer(x_hat)

    feat = {'x_hat': x_hat_q.z, 'spkr': spkr, 'pitch': None, 'energy': None}
    #Generator
    generator_buf = _generator.init_buffers(1, _device)
    wav, generator_buf = _generator(generator_buf, is_inference=True, **feat)

    out_audio = wav.squeeze()
    out_audio = out_audio * MAX_WAV_VALUE
    out_audio = out_audio.cpu().numpy().astype('int16')

    if out_file is not None:
        wavfile.write(out_file, sample_rate, out_audio)
    else:
        return out_audio

@torch.no_grad()
def process_wav_stream(wav_file, ref_wav_file=None, out_file=None, sample_rate:int=16000, chunk_size:int=200):
    """
    Args:
        wav_file: Path to audio wav file
        sample_rate: target sample rate
    
    returns:
        2D BNF vector (H, T)
    """
    audio, sr = librosa.load(wav_file, sr=sample_rate)
    # audio = audio / MAX_WAV_VALUE
    audio = torch.from_numpy(audio).to(_device)
    original_len = len(audio)

    #Spkr Embed
    if ref_wav_file is None:
        embed = get_anon_embed(wav_file)
    else:
        embed = get_speaker_embed(ref_wav_file)
    spkr = torch.from_numpy(embed).to(_device).unsqueeze(0)

    #create chunks
    n_chunk = chunk_size * 16
    if len(audio) % n_chunk != 0:
        pad_len = n_chunk - (len(audio) % n_chunk)
        audio = torch.nn.functional.pad(audio, (0, pad_len))

    audio_chunks = torch.split(audio, n_chunk)

    # print(_device)
    if _device == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    outputs = []
    times = []
    with torch.inference_mode():
        encoder_buf = _encoder.init_buffers(1, _device)
        generator_buf = _generator.init_buffers(1, _device)
        quantizer_buf = _quantizer.init_buffers(1, _device)
        
        for chunk in audio_chunks:
            start = time.time()
            if _device == 'cuda':
                starter.record()
            x_hat, encoder_buf = _encoder(
                chunk.unsqueeze(0).unsqueeze(0), 
                encoder_buf
            )

            x_hat_q, quantizer_buf = _quantizer(x_hat, quantizer_buf)

            feat = {'x_hat': x_hat_q.z, 'spkr': spkr, 'pitch': None, 'energy': None}
            
            out_audio, generator_buf = _generator(generator_buf, is_inference=True, **feat)
            out_audio = out_audio * MAX_WAV_VALUE
            outputs.append(out_audio)
            if _device == 'cuda':
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                times.append(curr_time)
            else:
                times.append(time.time() - start)
            
    
    # concatenate outputs
    outputs = torch.cat(outputs, dim=2)
    # Calculate RTF
    avg_time = np.mean(times)
    # print(avg_time)
    rtf = avg_time / (n_chunk / sr) 
    # calculate e2e latency
    e2e_latency = ((n_chunk / sr) + avg_time) * 1000
    # remove padding
    outputs = outputs[:, :, :original_len].squeeze()

    out_audio = outputs.cpu().numpy().astype('int16')
    if out_file is not None:
        wavfile.write(out_file, sample_rate, out_audio)
        return rtf, e2e_latency
    else:
        return out_audio, rtf, e2e_latency