import torch
import torch.nn as nn
from speechbrain.nnet.losses import bce_loss
from speechbrain.lobes.models.FastSpeech2 import SSIMLoss


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


class LossPED(nn.Module):
    """Loss Computation
    Arguments
    ---------
    log_scale_durations: bool
       applies logarithm to target durations
    duration_loss_weight: int
       weight for the duration loss
    pitch_loss_weight: int
       weight for the pitch loss
    energy_loss_weight: int
       weight for the energy loss
    """

    def __init__(
        self, h
    ):
        super().__init__()

        self.pitch_loss = nn.MSELoss()
        self.energy_loss = nn.MSELoss()
        self.pitch_loss_weight = h.pitch_loss_weight
        self.energy_loss_weight = h.energy_loss_weight

    def forward(self, predicted_pitch, average_pitch, 
                predicted_energy, average_energy):
        """Computes the value of the loss function and updates stats
        Arguments
        ---------
        predictions: tuple
            model predictions
        targets: tuple
            ground truth data
        Returns
        -------
        loss: torch.Tensor
            the loss value
        """

        predicted_pitch = predicted_pitch.squeeze(-1)
        predicted_energy = predicted_energy.squeeze(-1)

        target_pitch = average_pitch.squeeze(-1)
        target_energy = average_energy.squeeze(-1)

        # change this to perform batch level using padding mask
        pitch_loss = self.pitch_loss(
            predicted_pitch,
            target_pitch.to(torch.float32),
        )
        energy_loss = self.energy_loss(
            predicted_energy,
            target_energy.to(torch.float32),
        )
    
        total_loss = (
            pitch_loss * self.pitch_loss_weight
            + energy_loss * self.energy_loss_weight
        )
        
        loss = {
            "total_loss": total_loss,
            "pitch_loss": pitch_loss * self.pitch_loss_weight,
            "energy_loss": energy_loss * self.energy_loss_weight,
        }
        return loss