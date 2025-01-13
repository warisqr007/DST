"""BNF Quantizer related modules."""

from distutils.command.config import config
import imp
import logging
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize


class Quantizer(torch.nn.Module):
    """BNF Quantizer module.

    """

    def __init__(self, hdim=512, codebook_size=1024, kmeans_iters=100, commitment_cost=1, use_cosine_sim=True):
        # initialize base classes
        torch.nn.Module.__init__(self)
        
        self.vq = VectorQuantize(
                    dim = hdim,
                    codebook_size =  codebook_size,
                    kmeans_iters = kmeans_iters,
                    use_cosine_sim = use_cosine_sim   # set this to True
                )
        self.commitment_cost = commitment_cost

    def forward(self, xs, *args, **kwargs):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of padded acoustic features (B, Tmax, idim).
            ilens (LongTensor): Batch of lengths of each input batch (B,).

        Returns:
            Tensor: Loss value.

        """

        ##### Vector quantize BNFs ######
        xs_quantized, indices, commit_loss = self.vq(xs)
        #print(f'xs : {xs_quantized.shape} \n prosody_vec : {prosody_vec.shape} \nIndices: {indices.shape}')

        loss = self.commitment_cost*commit_loss

        return loss, commit_loss

    def inference(self, x, *args, **kwargs):
        """Generate the sequence of features given the sequences of acoustic features.

        Args:
            x (Tensor): Input sequence of acoustic features (T, idim).
            inference_args (Namespace):
                - threshold (float): Threshold in inference.
                - minlenratio (float): Minimum length ratio in inference.
                - maxlenratio (float): Maximum length ratio in inference.

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Encoder-decoder (source) attention weights (#layers, #heads, L, T).

        """
        

        # forward encoder
        x = x.unsqueeze(0)
        x_quantized, indices, _ = self.vq(x)

        return x_quantized.squeeze(0), indices.squeeze(0)