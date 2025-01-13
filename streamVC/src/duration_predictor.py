from torch import nn
from speechbrain.nnet import CNN, linear
from speechbrain.nnet.normalization import LayerNorm

from src.causal_conv import CausalConv


class DurationPredictor(nn.Module):
    """Duration predictor layer
    Arguments
    ---------
    in_channels: int
       input feature dimension for convolution layers
    out_channels: int
       output feature dimension for convolution layers
    kernel_size: int
       duration predictor convolution kernal size
    dropout: float
       dropout probability, 0 by default
    Example
    -------
    >>> from speechbrain.lobes.models.FastSpeech2 import FastSpeech2
    >>> duration_predictor_layer = DurationPredictor(in_channels=384, out_channels=384, kernel_size=3)
    >>> x = torch.randn(3, 400, 384)
    >>> mask = torch.ones(3, 400, 384)
    >>> y = duration_predictor_layer(x, mask)
    >>> y.shape
    torch.Size([3, 400, 1])
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, dropout=0.0
    ):
        super().__init__()
        self.conv1 = CausalConv(
            in_channel=in_channels,
            out_channel=out_channels,
            kernel_size=kernel_size,
            stride=1
        )
        self.conv2 = CausalConv(
            in_channel=out_channels,
            out_channel=out_channels,
            kernel_size=kernel_size,
            stride=1
        )
        
        self.linear = CausalConv(
            in_channel=out_channels,
            out_channel=1,
            kernel_size=1,
            stride=1
        )
        
        self.ln1 = LayerNorm(out_channels)
        self.ln2 = LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def init_buffers(self, batch_size, device):
        conv1_buf = self.conv1.init_ctx_buf(batch_size, device)
        conv2_buf = self.conv2.init_ctx_buf(batch_size, device)
        linear_buf = self.linear.init_ctx_buf(batch_size, device)
        ctx_buf = conv1_buf, conv2_buf, linear_buf
        return ctx_buf
    
    def forward(self, x, ctx_buf):
        """Computes the forward pass
        Arguments
        ---------
        x: torch.Tensor
            a (batch, time_steps, features) input tensor
        x_mask: torch.Tensor
            mask of input tensor
        Returns
        -------
        output: torch.Tensor
            the duration predictor outputs
        """
        conv1_buf, conv2_buf, linear_buf = ctx_buf

        x, conv1_buf = self.conv1(x, conv1_buf)
        x = self.relu(x)
        x = self.ln1(x.transpose(1,2)).to(x.dtype)
        x = self.dropout1(x)

        x, conv2_buf = self.conv2(x.transpose(1,2), conv2_buf)
        x = self.relu(x)
        x = self.ln2(x.transpose(1,2)).to(x.dtype)
        x = self.dropout2(x)

        x, linear_buf = self.linear(x.transpose(1,2), linear_buf)

        ctx_buf = conv1_buf, conv2_buf, linear_buf
        return x, ctx_buf
    
    def remove_weight_norm(self):
        self.conv1.remove_weight_norm()
        self.conv2.remove_weight_norm()
        self.linear.remove_weight_norm()