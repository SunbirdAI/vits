# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech





## Pre-requisites
0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`

## Training Exmaple
```
#Set up the config necessary
Example

config_file = """

config = {
    "model_dir": "/kaggle/working/best",
    "multispeaker": False,
    "mms_checkpoint": False,
    "ckpt_dir": None,
    "device": "cuda",
    "gcp_access": "secrets/srvc_acct.json",
    "drive_access": "/path/to/access/json or token",
    "vertex": {
        "gcp_project": "sb-gcp-project-01",
        "bucket_name": "ali_speech_experiments",
        "gcp_region": "europe-west6",
        "app_name": "train_tts", #according to //// format
        "prebuilt_docker_image": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-7:latest",
        "package_application_dir":"training",
        "source_package_file_name": "{}/dist/trainer-0.1.tar.gz", #root_dir same as package_application_dir
        "python_package_gcs_uri": "{}/pytorch-on-gcp/{}/train/python_package/trainer-0.1.tar.gz", #bucket_name app_name
        "python_module_name": "training.run", #To run?
        "requirements": [
            "Cython==0.29.21",
            "gdown",
            "google-cloud-storage",
            "librosa==0.8.0",
            "matplotlib==3.3.1",
            "numpy==1.18.5",
            "phonemizer==2.2.1",
            "scipy==1.5.2",
            #tensorboard==2.3.0
            "torch==1.6.0",
            "torchvision==0.7.0",
            "Unidecode==1.1.1"
        ]
        },
    "train": {
        "log_interval": 100,
        "eval_interval": 800,
        "seed": 1234,
        "epochs": 1000,
        "learning_rate": 2e-4,
        "betas": [0.8, 0.99],
        "eps": 1e-9,
        "batch_size": 32,
        "fp16_run": True,
        "lr_decay": 0.999875,
        "segment_size": 8192,
        "init_lr_ratio": 1,
        "warmup_epochs": 0,
        "c_mel": 45,
        "c_kl": 1.0
    },
    "data": {
        "balance":False,
        "download": False,
        "ogg_to_wav":False,
        "build_csv": False,
        "data_sources": [ #Ensure all datasets are in zip files
            ("gdrive", "1OTBFZCKv_RRIUWw7JaBIMPLkanEcWPJ2"),
            #https://drive.google.com/file/d/1OTBFZCKv_RRIUWw7JaBIMPLkanEcWPJ2/view?usp=drive_link
            #("bucket", "speech_collection_bucket" ,"VALIDATED/acholi-validated.zip")
            #("bucket", "speech_collection_bucket" ,"VALIDATED/lugbara-validated.zip")
            #("bucket", "speech_collection_bucket" ,"VALIDATED/luganda-validated.zip")
            #("bucket", "speech_collection_bucket" ,"VALIDATED/runyankole-validated.zip")
            #("bucket", "speech_collection_bucket" ,"VALIDATED/ateso-validated.zip")
            #("bucket", "speech_collection_bucket" ,"VALIDATED/english-validated.zip")
        ],
        "language": "luganda",
        "lang_iso": "lug",
        "reference_file":["/kaggle/temp/vits/training/training_files/Prompt-English.csv", "/kaggle/temp/vits/training/training_files/Prompt-Luganda.csv"],
        "training_files":"/kaggle/temp/lug_eng_csv/eng_lug_train.csv",
        "validation_files":"/kaggle/temp/lug_eng_csv/eng_lug_test.csv",
        "data_root_dir": "/kaggle/temp/datasets",
        "text_cleaners":["custom_cleaners"],
        "custom_cleaner_regex": ":!\?>\.;,",
        "max_wav_value": 32768.0,
        "sampling_rate": 22500,
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mel_channels": 80,
        "mel_fmin": 0.0,
        "mel_fmax": None,
        "add_blank": True,
        "n_speakers": 0,
        "cleaned_text": True
    },
    "model": {
        "vocab_file": "/kaggle/temp/eng_lug_pretrained/vocab.txt",
        "g_checkpoint_path": "/kaggle/temp/eng_lug_pretrained/G_eng_lug.pth",
        "d_checkpoint_path": "/kaggle/temp/eng_lug_pretrained/D_eng_lug.pth",
        "inter_channels": 192,
        "hidden_channels": 192,
        "filter_channels": 768,
        "n_heads": 2,
        "n_layers": 6,
        "kernel_size": 3,
        "p_dropout": 0.1,
        "resblock": "1",
        "resblock_kernel_sizes": [3,7,11],
        "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
        "upsample_rates": [8,8,2,2],
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [16,16,4,4],
        "n_layers_q": 3,
        "use_spectral_norm": False,
        "gin_channels": 256
    }
}


config["vocab_file"] =  f"{ config['ckpt_dir'] }/vocab.txt"
config["config_file"] =  f"{ config['ckpt_dir'] }/config.json"


"""


with open("vits/training/train_config.py", "w") as tfd:
    tfd.write(config_file)
%cd vits/training
!python fix_torch.py
%cd ../..

%cd vits/training
!python run.py
%cd ../..
```

## Inference Example
```
from training.hf_wrapper import VITSInfereceAdapterModel
# Define constants and prerequisites
REPO_NAME = "username/repo-name-on-huggingface"
G_NET_PATH = "path_to_your_G_eng_lug.pth_in_repo"  # Update this path as per your repo
VOCAB_PATH = "vocab.txt"
CONFIG_PATH = "config.json"  # Optional
TEXT_TO_ENCODE = "Hello, world!"

# Use the from_pretrained class method to initialize the model
model = VITSInfereceAdapterModel.from_pretrained(REPO_NAME, G_NET_PATH, VOCAB_PATH, CONFIG_PATH)

# Encode a piece of text using the model
encoded_text = model.encode_text(TEXT_TO_ENCODE)

# Print or process the encoded text
print(encoded_text)
```
