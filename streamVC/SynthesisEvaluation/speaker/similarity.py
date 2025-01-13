import numpy as np
import torch
from resemblyzer import preprocess_wav, VoiceEncoder

_model = None # type: VoiceEncoder
_device = None # type: torch.device


def load_model(device=None):
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
    global _model, _device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _device = torch.device(device)
    elif isinstance(device, str):
        _device = torch.device(device)
        
    _model = VoiceEncoder().to(device)
    _model.eval()
    print("Loaded Speaker encoder.")


def is_loaded():
    return _model is not None

def utterance_similarity(wav_file, wav_file_ref):
    """
    Args:
        wav_file: Path to audio wav file
        sample_rate: target sample rate
    
    returns:
        2D BNF vector (H, T)
    """ 
    wav = preprocess_wav(wav_file)
    wav_ref = preprocess_wav(wav_file_ref)
    embed = _model.embed_utterance(wav)
    embed_ref = _model.embed_utterance(wav_ref)
    
    utt_sim = embed @ embed_ref
    return utt_sim
