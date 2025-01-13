from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from vector_quantize_pytorch import FSQ, GroupedResidualFSQ

from src.causal_conv import CausalConv, CausalConvTranspose1D
from src.causal_resblock import ResBlock

@dataclass
class FSQResult:
    z: torch.Tensor
    codes: torch.Tensor
    latents: torch.Tensor


class FiniteScalarQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        levels: list[int] = [8, 6, 5],  # target size 2^8, actual size 240
    ):
        super().__init__()

        self.fsq = FSQ(
            dim=input_dim,
            levels=levels,
        )

    def forward(self, z) -> FSQResult:
        quantized, indices = self.fsq(z.mT)
        result = FSQResult(
            z=quantized.mT,
            codes=indices,
            latents=z,
        )
        return result

    def encode(self, z):
        _, indices = self.fsq(z.mT)
        return indices

    def decode(self, indices: torch.Tensor):
        z_q = self.fsq.indices_to_codes(indices)
        return z_q.mT


class DownsampleFiniteScalarQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        levels: list[int] = [8, 5, 5, 5],  # Approximate 2**10
        downsample_factor: tuple[int] = (2, 2),
        downsample_dims: tuple[int] | None = None,
    ):
        super().__init__()

        if downsample_dims is None:
            downsample_dims = [input_dim for _ in range(len(downsample_factor))]

        all_dims = (input_dim,) + tuple(downsample_dims)

        self.fsq = FSQ(
            dim=all_dims[-1],
            levels=levels,
        )

        self.downsample_factor = downsample_factor
        self.downsample_dims = downsample_dims

        # Downsample
        self.downs = nn.ModuleList()
        for idx, factor in enumerate(downsample_factor):
            self.downs.append(
                CausalConv(all_dims[idx], 
                           all_dims[idx + 1], 
                           factor, factor)
                )

        self.down_resblocks = nn.ModuleList()
        for i in range(len(downsample_factor)):
            self.down_resblocks.append(ResBlock(None, all_dims[idx + 1], 7, 0))


        # Upsample
        self.ups = nn.ModuleList()
        for idx, factor in reversed(list(enumerate(downsample_factor))):
            self.ups.append(
                CausalConvTranspose1D(all_dims[idx + 1], 
                                      all_dims[idx], 
                                      factor, factor)
                )
            
        self.ups_resblocks = nn.ModuleList()
        for idx, factor in reversed(list(enumerate(downsample_factor))):
            self.ups_resblocks.append(ResBlock(None, all_dims[idx], 7, 0))

    def init_buffers(self, batch_size, device):
        down_buf = []
        for i in range(len(self.downs)):
            ctx_buf = self.downs[i].init_ctx_buf(batch_size, device)
            down_buf.append(ctx_buf)

            res_buf = self.down_resblocks[i].init_ctx_buf(batch_size, device)
            down_buf.append(res_buf)

        up_buf = []
        for i in range(len(self.ups)):
            ctx_buf = self.ups[i].init_ctx_buf(batch_size, device)
            up_buf.append(ctx_buf)

            res_buf = self.ups_resblocks[i].init_ctx_buf(batch_size, device)
            up_buf.append(res_buf)
        
        return down_buf, up_buf
        
    def remove_weight_norm(self):
        for down in self.downs:
            down.remove_weight_norm()
        for up in self.ups:
            up.remove_weight_norm()
        for resblock in self.down_resblocks:
            resblock.remove_weight_norm()
        for resblock in self.ups_resblocks:
            resblock.remove_weight_norm()


    def downsample(self, z, down_buf):
        for i, (down, resblock) in enumerate(zip(self.downs, self.down_resblocks)):
            z, down_buf[2*i] = down(z, down_buf[2*i])
            z, down_buf[2*i + 1] = resblock(z, down_buf[2*i + 1])
        
        return z, down_buf

    def upsample(self, z, up_buf):
        for i, (up, resblock) in enumerate(zip(self.ups, self.ups_resblocks)):
            z, up_buf[2*i] = up(z, up_buf[2*i])
            z, up_buf[2*i + 1] = resblock(z, up_buf[2*i + 1])

        return z, up_buf


    def forward(self, z, ctx_buf=None) -> FSQResult:
        if ctx_buf is None:
            down_buf, up_buf = self.init_buffers(z.shape[0], z.device)
        else:
            down_buf, up_buf = ctx_buf

        # Downsample
        z, down_buf = self.downsample(z, down_buf)

        # Quantize
        quantized, indices = self.fsq(z.mT)
        result = FSQResult(
            z=quantized.mT,
            codes=indices,
            latents=z,
        )

        # Upsample
        result.z, up_buf = self.upsample(result.z, up_buf)

        return result, (down_buf, up_buf)

    def encode(self, z, down_buf=None):
        if down_buf is None:
            down_buf, _ = self.init_buffers(z.shape[0], z.device)
        
        z, down_buf = self.downsample(z, down_buf)
        _, indices = self.fsq(z.mT)
        return indices, down_buf
    
    def decode(self, indices: torch.Tensor, up_buf=None):
        if up_buf is None:
            _, up_buf = self.init_buffers(z.shape[0], z.device)

        z_q = self.fsq.indices_to_codes(indices)
        z_q, up_buf = self.upsample(z_q.mT, up_buf)
        return z_q, up_buf