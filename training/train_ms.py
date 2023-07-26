import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from text.mappers import TextMapper, preprocess_char
from misc import filter_corrupt_files, download_and_extract_drive_file, download_blob

import commons
import utils
from data_utils import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

from data_utils import verify_audio_dir

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '80000'

  hps = utils.get_hparams()
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, config,device="cpu", g_checkpoint_path = None, d_checkpoint_path = None):
  global global_step

  corrupt_list = verify_audio_dir(config["data"]["data_root_dir"], file_extension=".wav")

  # try:
  #   assert len(corrupt_list) == 0
  # except:
  #   print(corrupt_list)
  #   raise ValueError("Handle corrupt files first")

  if config["data"]["download"]:
    for data_source in config["data"]["data_sources"]:
        if data_source[0] == "gdrive":
          file_id = data_source[1]
          download_and_extract_drive_file(file_id,config["data"]["data_root_dir"] )
        elif data_source[0] == "bucket":
          bucket_name = data_source[1]
          blob_name = data_source[2]
          download_blob(bucket_name,blob_name, config["data"]["data_root_dir"])

  filter_corrupt_files(config["data"]["training_files"], "|")
  filter_corrupt_files(config["data"]["validation_files"], "|")


  logger = utils.get_logger(config["model_dir"])
  logger.info(config)
  utils.check_git_hash(config["model_dir"])
  #writer = SummaryWriter(log_dir=config["model_dir"])
  #writer_eval = SummaryWriter(log_dir=os.path.join(config["model_dir"], "eval"))

  
  torch.manual_seed(config["train"]["seed"])
  #torch.cuda.set_device(rank)
  text_mapper = TextMapper(config["model"]["vocab_file"])
  train_dataset = TextAudioSpeakerLoader(config["data"]["training_files"], config["data"], text_mapper)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      config["train"]["batch_size"],
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=1,
      rank=0,
      shuffle=True)
  collate_fn = TextAudioSpeakerCollate()
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)

  eval_dataset = TextAudioSpeakerLoader(config["data"]["validation_files"], config["data"], text_mapper)
  eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False,
      batch_size=config["train"]["batch_size"], pin_memory=True,
      drop_last=False, collate_fn=collate_fn)

  net_g = SynthesizerTrn(
      len(symbols),
      config["data"]["filter_length"] // 2 + 1,
      config["train"]["segment_size"] // config["data"]["hop_length"],
      n_speakers=config["data"]["n_speakers"],
      **config["model"]).to(device)
  net_d = MultiPeriodDiscriminator(config["model"]["use_spectral_norm"]).to(device)
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      config["train"]["learning_rate"], 
      betas=config["train"]["betas"], 
      eps=config["train"]["eps"])
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      config["train"]["learning_rate"], 
      betas=config["train"]["betas"], 
      eps=config["train"]["eps"])
  #net_g = DDP(net_g, device_ids=[rank])
  #net_d = DDP(net_d, device_ids=[rank])

  try:
    _, _, _, epoch_str = utils.load_checkpoint(g_checkpoint_path, net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(d_checkpoint_path, net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config["train"]["lr_decay"], last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config["train"]["lr_decay"], last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=config["train"]["fp16_run"])

  for epoch in range(epoch_str, config["train"]["epochs"] + 1):
    if rank==0:
      train_and_evaluate(config, epoch, config, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, None)
    else:
      train_and_evaluate(config, epoch, config, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
    scheduler_g.step()
    scheduler_d.step()


def train_and_evaluate(config, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  device = config["device"]  
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  #if writers is not None:
  #  writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()
  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(train_loader):
    x, x_lengths = x.to(device), x_lengths.to(device)
    spec, spec_lengths = spec.to(device), spec_lengths.to(device)
    y, y_lengths = y.to(device), y_lengths.to(device)
    speakers = speakers.to(device)

    with autocast(enabled=config["train"]["fp16_run"]):
      y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, speakers)

      mel = spec_to_mel_torch(
          spec, 
          config["data"]["filter_length"], 
          config["data"]["n_mel_channels"], 
          config["data"]["sampling_rate"],
          config["data"]["mel_fmin"], 
          config["data"]["mel_fmax"])
      y_mel = commons.slice_segments(mel, ids_slice, config["train"]["segment_size"] // config["data"]["hop_length"])
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          config["data"]["filter_length"], 
          config["data"]["n_mel_channels"], 
          config["data"]["sampling_rate"], 
          config["data"]["hop_length"], 
          config["data"]["win_length"], 
          config["data"]["mel_fmin"], 
          config["data"]["mel_fmax"]
      )

      y = commons.slice_segments(y, ids_slice * config["data"]["hop_length"], config["train"]["segment_size"]) # slice 

      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=config["train"]["fp16_run"]):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      with autocast(enabled=False):
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * config["train"]["c_mel"]
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config["train"]["c_kl"]

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if global_step % config["train"]["log_interval"] == 0:
      lr = optim_g.param_groups[0]['lr']
      losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
      logger.info('Train Epoch: {} [{:.0f}%]'.format(
        epoch,
        100. * batch_idx / len(train_loader)))
      logger.info([x.item() for x in losses] + [global_step, lr])
      
      scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
      scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})

      scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
      scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
      scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
      image_dict = { 
          "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
          "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
          "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
          "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
      }
      # utils.summarize(
      #   writer=writer,
      #   global_step=global_step, 
      #   images=image_dict,
      #   scalars=scalar_dict)

    if global_step % config["train"]["eval_interval"] == 0:
      evaluate(config, net_g, eval_loader, None)
      utils.save_checkpoint(net_g, optim_g, config["train"]["learning_rate"], epoch, os.path.join(config["model_dir"], "G_{}.pth".format(global_step)))
      utils.save_checkpoint(net_d, optim_d, config["train"]["learning_rate"], epoch, os.path.join(config["model_dir"], "D_{}.pth".format(global_step)))
  global_step += 1

  logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(config, generator, eval_loader, writer_eval):
    generator.eval()
    device = config["device"]
    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(eval_loader):
        x, x_lengths = x.to(device), x_lengths.to(device)
        spec, spec_lengths = spec.to(device), spec_lengths.to(device)
        y, y_lengths = y.to(device), y_lengths.to(device)
        speakers = speakers.to(device)

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        speakers = speakers[:1]
        break
      y_hat, attn, mask, *_ = generator.infer(x, x_lengths, speakers, max_len=1000)
      y_hat_lengths = mask.sum([1,2]).long() * config["data"]["hop_length"]

      mel = spec_to_mel_torch(
        spec, 
        config["data"]["filter_length"], 
        config["data"]["n_mel_channels"], 
        config["data"]["sampling_rate"],
        config["data"]["mel_fmin"], 
        config["data"]["mel_fmax"])
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        config["data"]["filter_length"],
        config["data"]["n_mel_channels"],
        config["data"]["sampling_rate"],
        config["data"]["hop_length"],
        config["data"]["win_length"],
        config["data"]["mel_fmin"],
        config["data"]["mel_fmax"]
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    # utils.summarize(
    #   writer=writer_eval,
    #   global_step=global_step, 
    #   images=image_dict,
    #   audios=audio_dict,
    #   audio_sampling_rate=config["data"]["sampling_rate"]
    # )
    generator.train()

                           
if __name__ == "__main__":
  main()
