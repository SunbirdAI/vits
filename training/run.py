from train_config import config
import os
import subprocess
from text.mappers import TextMapper, preprocess_char
from train import run as train_single
from train_ms import run as train_multi
import numpy as np
if config["multispeaker"]:
    train_multi(0, 0,  config, g_checkpoint_path = None, d_checkpoint_path = None)
else:
    train_single(0, 0,  config, g_checkpoint_path = None, d_checkpoint_path = None)
    


# vocab_file = f"{ckpt_dir}/vocab.txt"
# config_file = f"{ckpt_dir}/config.json"
# assert os.path.isfile(config_file), f"{config_file} doesn't exist"
# hps = utils.get_hparams_from_file(config_file)
# text_mapper = TextMapper(vocab_file)
# net_g = SynthesizerTrn(
#     len(text_mapper.symbols),
#     hps.data.filter_length // 2 + 1,
#     hps.train.segment_size // hps.data.hop_length,
#     **hps.model)
# net_g.to(device)
#
# g_pth = f"/content/drive/MyDrive/models/vits/G_9000.pth" enviroment_variable this in the docker file
# print(f"load {g_pth}")
#
# _ = utils.load_checkpoint(g_pth, net_g, None)
#  acholi_multi_val_n_test.csv to acholi_multi_val_n_test.csv
#  acholi_multi_train.csv to acholi_multi_train.csv
#
#
# Then the config (find in cell below)
#
#
# Modify train and train_ms to take the checkpoints from env variables not the highest value, config.json, vocab_file_path as env variables as well
#
# Wandb from scalar_dict NO tensorboard
#
# Make sure to modify text/cleaners.py to match language (acholi has no q). Also modify the TextMapper and Syn model accordingly.
#
# Customize embedding layers when using a Syn checkpoint to another with a different size of embedding layer
