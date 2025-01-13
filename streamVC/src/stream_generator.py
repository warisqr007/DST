import torch
import torch.nn.functional as F
import torch.nn as nn

from src.utils import init_weights
from src.causal_conv import CausalConv, CausalConvTranspose1D
from src.causal_resblock import ResBlock
from src.speaker_adapter import SpeakerAdapter

LRELU_SLOPE = 0.1

class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.up_buf_len = 1

        resblock = ResBlock
        
        self.conv_pre = CausalConv(getattr(h, "model_in_dim", 128), h.upsample_initial_channel, 7, 1)

        self.ups = nn.ModuleList()
        self.spk_adapters = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                CausalConvTranspose1D(h.upsample_initial_channel // (2 ** i), 
                                      h.upsample_initial_channel // (2 ** (i + 1)), 
                                      k, u))
            if i!=0:
                self.spk_adapters.append(SpeakerAdapter(h.embed_in_dim, h.upsample_initial_channel // (2 ** i)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, 0, d))

        self.conv_post = CausalConv(ch, 1, 7, 1)
        for l in self.ups:
            l.init_weights()
        self.conv_post.layer.apply(init_weights)

    def init_buffers(self, batch_size, device):
        res_buf = []
        for i in range(self.num_upsamples):
            for j in range(self.num_kernels):
                ctx_buf = self.resblocks[i * self.num_kernels + j].init_ctx_buf(batch_size, device)
                res_buf.append(ctx_buf)
        up_buf = []
        for i, (u, k) in enumerate(zip(self.h.upsample_rates, self.h.upsample_kernel_sizes)):
            ctx_buf = self.ups[i].init_ctx_buf(batch_size, device)
            up_buf.append(ctx_buf)

        pre_conv_buf = self.conv_pre.init_ctx_buf(batch_size, device)
        post_conv_buf = self.conv_post.init_ctx_buf(batch_size, device)
        buffers = pre_conv_buf, res_buf, up_buf, post_conv_buf
        return buffers
    
    def forward(self, x, embeds, buffers):
        pre_conv_buf, res_buf, up_buf, post_conv_buf = buffers
        #Add post conv buff
        x, pre_conv_buf = self.conv_pre(x, pre_conv_buf)
        for i in range(self.num_upsamples):
            if i!=0:
                x = self.spk_adapters[i-1](x, embeds)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x, up_buf[i] = self.ups[i](x, up_buf[i])
            xs = None
            for j in range(self.num_kernels):
                ctx_buf = res_buf[i * self.num_kernels + j]
                if xs is None:
                    xs, ctx_buf = self.resblocks[i * self.num_kernels + j](x, ctx_buf)
                else:
                    xs_, ctx_buf = self.resblocks[i * self.num_kernels + j](x, ctx_buf)
                    xs += xs_
                res_buf[i * self.num_kernels + j] = ctx_buf
            x = xs / self.num_kernels
        x = F.leaky_relu(x)

        #Add post conv buff
        x, post_conv_buf = self.conv_post(x, post_conv_buf)
        x = torch.tanh(x)

        buffers = pre_conv_buf, res_buf, up_buf, post_conv_buf
        return x, buffers

    def remove_weight_norm(self):
        for l in self.ups:
            l.remove_weight_norm()
        for l in self.resblocks:
            l.remove_weight_norm()
        self.conv_pre.remove_weight_norm()
        self.conv_post.remove_weight_norm()