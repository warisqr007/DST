import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from src.duration_predictor import DurationPredictor
from src.stream_generator import Generator
from src.causal_conv import CausalConv
from src.speaker_adapter import SpeakerAdapter


LRELU_SLOPE = 0.1

class CodeGenerator(Generator):
    def __init__(self, h):
        super().__init__(h)
        
        self.h = h
        self.use_duration_pred = h.use_duration_pred
        self.use_only_native_for_gen = h.use_only_native_for_gen
        
        self.spk_adapter = SpeakerAdapter(h.embed_in_dim, h.upsample_initial_channel)
        
        self.pitchPred = DurationPredictor(
            in_channels=h.model_in_dim,
            out_channels=h.model_in_dim,
            kernel_size=h.dur_pred_kernel_size,
            dropout=h.variance_predictor_dropout,
        )
        self.energyPred = DurationPredictor(
            in_channels=h.model_in_dim,
            out_channels=h.model_in_dim,
            kernel_size=h.dur_pred_kernel_size,
            dropout=h.variance_predictor_dropout,
        )
        self.pitchEmbed = CausalConv(
            in_channel=1,
            out_channel=h.model_in_dim,
            kernel_size=h.pitch_pred_kernel_size,
            stride=1
        )

        self.energyEmbed = CausalConv(
            in_channel=1,
            out_channel=h.model_in_dim,
            kernel_size=h.energy_pred_kernel_size,
            stride=1
        )

    def _average_over_durations(self, values, durs):
        """Average values over durations.
        Arguments
        ---------
        values: torch.Tensor
            shape: [B, 1, T_de]
        durs: torch.Tensor
            shape: [B, T_en]
        Returns
        ---------
        avg: torch.Tensor
            shape: [B, 1, T_en]
        """
        avg_batch = []
        for i, (d, v) in enumerate(zip(durs, values)):
            v = v.unsqueeze(0)
            durs_cums_ends = torch.cumsum(d.unsqueeze(0), dim=1).long()
            durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
            values_nonzero_cums = torch.nn.functional.pad(
                torch.cumsum(v != 0.0, dim=2), (1, 0)
            )
            values_cums = torch.nn.functional.pad(torch.cumsum(v, dim=2), (1, 0))

            bs, length = durs_cums_ends.size()
            n_formants = v.size(1)
            dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, length)
            dce = durs_cums_ends[:, None, :].expand(bs, n_formants, length)

            values_sums = (
                torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)
            ).float()
            values_nelems = (
                torch.gather(values_nonzero_cums, 2, dce)
                - torch.gather(values_nonzero_cums, 2, dcs)
            ).float()

            avg = torch.where(
                values_nelems == 0.0, values_nelems, values_sums / values_nelems
            )
            avg_batch.append(avg.squeeze(0).repeat_interleave(d, dim=-1))

        return torch.stack(avg_batch, dim=0)

    def _get_durations(self, tokens):
        durations = [
            torch.unique_consecutive(tokens[i], return_counts=True)[1]
            for i in range(len(tokens))
        ]
        # padded_durations = torch.nn.utils.rnn.pad_sequence(
        #     durations, batch_first=True, padding_value=0.0,
        # )
        return durations
    
    def init_buffers(self, batch_size, device):
        pre_conv_buf, res_buf, up_buf, post_conv_buf = super().init_buffers(batch_size, device)
        code_gen_buf = []
        code_gen_buf.append(self.pitchPred.init_buffers(batch_size, device))
        code_gen_buf.append(self.pitchEmbed.init_ctx_buf(batch_size, device))
        code_gen_buf.append(self.energyPred.init_buffers(batch_size, device))
        code_gen_buf.append(self.energyEmbed.init_ctx_buf(batch_size, device))

        return code_gen_buf, pre_conv_buf, res_buf, up_buf, post_conv_buf


    def forward(self, ctx_buf, is_inference=False, **kwargs):
        
        code_gen_buf, pre_conv_buf, res_buf, up_buf, post_conv_buf = ctx_buf

        x = kwargs['x_hat']
        # Perturbation
        # x = x + torch.randn_like(x)
        durations = None
        if not is_inference:
            durations = self._get_durations(kwargs['tokens'])

        # Add speaker info
        embeds = kwargs['spkr']
        token_feats = self.spk_adapter(x, embeds)


        # pitch predictor
        avg_pitch = None
        predict_pitch, code_gen_buf[0] = self.pitchPred(token_feats, code_gen_buf[0])
        # use a pitch rate to adjust the pitch
        predict_pitch = predict_pitch * self.h.pitch_rate

        if kwargs['pitch'] is not None:
            avg_pitch = self._average_over_durations(kwargs['pitch'].unsqueeze(1), durations)
            pitch, code_gen_buf[1] = self.pitchEmbed(avg_pitch, code_gen_buf[1])
            #avg_pitch = avg_pitch.permute(0, 2, 1)
        else:
            pitch, code_gen_buf[1] = self.pitchEmbed(predict_pitch, code_gen_buf[1]) #.permute(0, 2, 1))
        #pitch = pitch.permute(0, 2, 1)
        token_feats = token_feats.add(pitch)

        # energy predictor
        avg_energy = None
        predict_energy, code_gen_buf[2] = self.energyPred(token_feats, code_gen_buf[2])
        # use an energy rate to adjust the energy
        predict_energy = predict_energy * self.h.energy_rate
        if kwargs['energy'] is not None:
            avg_energy = self._average_over_durations(kwargs['energy'].unsqueeze(1), durations)
            energy, code_gen_buf[3] = self.energyEmbed(avg_energy, code_gen_buf[3])
            # avg_energy = avg_energy.permute(0, 2, 1)
        else:
            energy, code_gen_buf[3] = self.energyEmbed(predict_energy, code_gen_buf[3])#.permute(0, 2, 1))
        # energy = energy.permute(0, 2, 1)
        token_feats = token_feats.add(energy)

        if getattr(self.h, "use_variance_adapter", False):
            x = token_feats
        
        x, (pre_conv_buf, res_buf, up_buf, post_conv_buf) = super().forward(x, embeds, (pre_conv_buf, res_buf, up_buf, post_conv_buf))
        ctx_buf = code_gen_buf, pre_conv_buf, res_buf, up_buf, post_conv_buf

        if is_inference:
            return x, ctx_buf
        
        return x, ctx_buf, predict_pitch, avg_pitch, predict_energy, avg_energy,
    
    def remove_weight_norm(self):
        super().remove_weight_norm()
        self.pitchEmbed.remove_weight_norm()
        self.energyEmbed.remove_weight_norm()
        self.pitchPred.remove_weight_norm()
        self.energyPred.remove_weight_norm()