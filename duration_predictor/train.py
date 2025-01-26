import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', message='.*kernel_size exceeds volume extent.*')

import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from src.dataset import CodeDatasetPED as CodeDataset, mel_spectrogram, get_dataset_filelist
from src.encoder import Encoder
from src.quantizer.bnftocode import Quantizer
from src.code_generator import CodeGenerator
from src.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator 
from src.code_predictor import CodePredictor
from src.loss import feature_loss, generator_loss, discriminator_loss, LossPED
from src.stft_loss import MultiResolutionSTFTLoss
from src.utils import plot_spectrogram, scan_checkpoint, load_checkpoint, \
    save_checkpoint, build_env, AttrDict, get_fairseq_model, fairseq_loss, speaker_loss
from speaker import encoder as embedder

torch.backends.cudnn.benchmark = True


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            rank=rank,
            world_size=h.dist_config['world_size'] * h.num_gpus,
        )

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    encoder = Encoder(h).to(device)
    quantizer = Quantizer(hdim=h.upsample_initial_channel, codebook_size=h.vq_size).to(device)
    generator = CodeGenerator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    
    stft_criterion = MultiResolutionSTFTLoss(device, eval(h.mrd_resolutions))
    ped_criterion = LossPED(h)

    if h.use_speaker_loss:
        embedder.load_model(device)

    if h.use_fairseq_loss:
        fairseq_model = get_fairseq_model(h, device)

    if rank == 0:
        # print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    last_epoch = -1
    if cp_g is None or cp_do is None:
        state_dict_do = None

        # load pretrained encoder
        cp_g_e = scan_checkpoint(a.pretrained_model_path, 'g_')
        cp_do_e = scan_checkpoint(a.pretrained_model_path, 'do_')
        cp_vq = scan_checkpoint(a.pretrained_vq_path, 'vq_')
        if cp_g_e is None:
            raise ValueError("Pretrained encoder checkpoint not found")
        state_dict_g_e = load_checkpoint(cp_g_e, device)
        state_dict_do_e = load_checkpoint(cp_do_e, device)
        state_dict_vq = load_checkpoint(cp_vq, device)
        encoder.load_state_dict(state_dict_g_e['encoder'])
        generator.load_state_dict(state_dict_g_e['generator'])
        mpd.load_state_dict(state_dict_do_e['mpd'])
        msd.load_state_dict(state_dict_do_e['msd'])
        quantizer.load_state_dict(state_dict_vq['quantizer'])
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        encoder.load_state_dict(state_dict_g['encoder'])
        quantizer.load_state_dict(state_dict_g['quantizer'])
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        encoder = DistributedDataParallel(encoder, device_ids=[rank]).to(device)
        generator = DistributedDataParallel(
            generator,
            device_ids=[rank],
        ).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), 
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), h.learning_rate,
                                betas=[h.adam_b1, h.adam_b2])


    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h)

    trainset = CodeDataset(training_filelist, h.segment_size, h.code_hop_size, 
                           h.n_fft, h.num_mels, h.hop_size,
                           h.win_size, h.sampling_rate, h.fmin, h.fmax, h.km_path,
                           fmax_loss=h.fmax_for_loss,
                           device=device)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False, sampler=train_sampler,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True)

    if rank == 0:
        validset = CodeDataset(validation_filelist, h.segment_size, h.code_hop_size, 
                               h.n_fft, h.num_mels, h.hop_size,
                               h.win_size, h.sampling_rate, h.fmin, h.fmax, h.km_path,
                               False,
                               fmax_loss=h.fmax_for_loss, device=device)
        validation_loader = DataLoader(validset, num_workers=h.num_workers, shuffle=False, sampler=None,
                                       batch_size=h.batch_size, pin_memory=True, drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    encoder.eval()
    quantizer.eval()
    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch
            y = torch.autograd.Variable(y.to(device, non_blocking=False))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=False))
            y = y.unsqueeze(1)
            x = {k: torch.autograd.Variable(v.to(device, non_blocking=False)) for k, v in x.items()}
            
            # Streaming Encoder
            if hasattr(encoder, 'module'):
                encoder_buf = encoder.module.init_buffers(y.shape[0], y.device)
                x_hat, encoder_buf = encoder.module(y, encoder_buf)
            else:
                encoder_buf = encoder.init_buffers(y.shape[0], y.device)
                x_hat, encoder_buf = encoder(y, encoder_buf)
            

            # Quantizer
            x_hat_quantized, indices, _ = quantizer.vq(x_hat.transpose(1,2))
            
            
            # Streaming Decoder
            if hasattr(generator, 'module'):
                generator_buf = generator.module.init_buffers(y.shape[0], y.device)
                x['x_hat'] = x_hat_quantized.transpose(1,2).detach() #+ 0.05 * torch.randn_like(x_hat_quantized.transpose(1,2))
                y_g_hat, generator_buf, predict_pitch, avg_pitch, \
                    predict_energy, avg_energy = generator.module(generator_buf, **x)
            else:
                generator_buf = generator.init_buffers(y.shape[0], y.device)
                x['x_hat'] = x_hat_quantized.transpose(1,2).detach() #+ 0.05 * torch.randn_like(x_hat_quantized.transpose(1,2))
                y_g_hat, generator_buf, predict_pitch, avg_pitch, \
                    predict_energy, avg_energy = generator(generator_buf, **x)
            
            assert y_g_hat.shape == y.shape, f"Mismatch in vocoder output shape - {y_g_hat.shape} != {y.shape}"

            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                                          h.win_size, h.fmin, h.fmax_for_loss)

            # Discriminators
            optim_d.zero_grad()

            # MPD
            if hasattr(mpd, 'module'):
                y_df_hat_r, y_df_hat_g, _, _ = mpd.module(y, y_g_hat.detach())
            else:
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            if hasattr(msd, 'module'):
                y_ds_hat_r, y_ds_hat_g, _, _ = msd.module(y, y_g_hat.detach())
            else:
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 9
            
            #Multi-Resolution STFT Loss
            sc_loss, mag_loss = stft_criterion(y_g_hat.squeeze(1), y.squeeze(1))
            stft_loss = (sc_loss + mag_loss) * 18

            if hasattr(mpd, 'module'):
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd.module(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd.module(y, y_g_hat)
            else:
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

            ped_loss = ped_criterion(predict_pitch, avg_pitch, predict_energy, avg_energy)
            
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + stft_loss + 50 * ped_loss["total_loss"]

            if h.use_fairseq_loss:
                loss_f = 100000 * fairseq_loss(y_g_hat, y, fairseq_model)
                loss_gen_all += loss_f
            else:
                loss_f = torch.tensor(0.0)

            if h.use_speaker_loss:
                loss_spkr = 10 * speaker_loss(y_g_hat, x['spkr'], embedder)
                loss_gen_all += loss_spkr
            else:
                loss_spkr = torch.tensor(0.0)
                        
            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print(
                        'Steps : {:d},  Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, stft : {:4.3f}, F_loss : {:4.3f}, S_loss: {:4.3f}, ped : {:4.3f}, s/b :{:4.3f}'.format(
                            steps,
                            loss_gen_all,
                            mel_error,
                            stft_loss.item(),
                            loss_f.item(),
                            loss_spkr.item(),
                            ped_loss["total_loss"].item(),
                            time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict(),
                                     'quantizer': quantizer.state_dict(),
                                     'encoder': (encoder.module if h.num_gpus > 1 else encoder).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, {'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                                                      'msd': (msd.module if h.num_gpus > 1 else msd).state_dict(),
                                                      'optim_g': optim_g.state_dict(), 
                                                      'optim_d': optim_d.state_dict(),
                                                      'steps': steps, 'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    sw.add_scalar("training/fairseq_loss", loss_f, steps)
                    sw.add_scalar("training/speaker_loss", loss_spkr, steps)
                    
                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    encoder.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch
                            y = y.to(device, non_blocking=False)
                            x = {k: v.to(device, non_blocking=False) for k, v in x.items()}

                            if hasattr(encoder, 'module'):
                                encoder_buf = encoder.module.init_buffers(y.shape[0], y.device)
                                x_hat, encoder_buf = encoder.module(y.unsqueeze(1), encoder_buf)
                            else:
                                encoder_buf = encoder.init_buffers(y.shape[0], y.device)
                                x_hat, encoder_buf = encoder(y.unsqueeze(1), encoder_buf)

                            x_hat_quantized, indices, _ = quantizer.vq(x_hat.transpose(1,2))

                            if hasattr(generator, 'module'):
                                generator_buf = generator.module.init_buffers(y.shape[0], y.device)
                                x['x_hat'] = x_hat_quantized.transpose(1,2)
                                y_g_hat, generator_buf, predict_pitch, avg_pitch, \
                                    predict_energy, avg_energy = generator.module(generator_buf, **x)
                            else:
                                generator_buf = generator.init_buffers(y.shape[0], y.device)
                                x['x_hat'] = x_hat_quantized.transpose(1,2)
                                y_g_hat, generator_buf, predict_pitch, avg_pitch, \
                                    predict_energy, avg_energy = generator(generator_buf, **x)

                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=False))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(y_mel[0].cpu()), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat[:1].squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec[:1].squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)
                    generator.train()

            steps += 1
            if steps >= a.training_steps:
                break

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

    if rank == 0:
        print('Finished training')


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--checkpoint_path', default='experiments/base')
    parser.add_argument('-c', '--config', default='experiments/base/config.json')
    parser.add_argument('--pretrained_model_path', default='experiments/base/encoder')
    parser.add_argument('--pretrained_vq_path', default='experiments/base/vq')
    parser.add_argument('--training_epochs', default=6000, type=int)
    parser.add_argument('--training_steps', default=800000, type=int)
    parser.add_argument('--stdout_interval', default=20, type=int)
    parser.add_argument('--checkpoint_interval', default=50000, type=int)
    parser.add_argument('--summary_interval', default=50, type=int)
    parser.add_argument('--validation_interval', default=10, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    # build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
