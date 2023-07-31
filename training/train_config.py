config = {
    "model_dir": "best",
    "multispeaker": True,
    "mms_checkpoint": False,
    "ckpt_dir": None,
    "device": "cpu",
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
        "log_interval": 200,
        "eval_interval": 1000,
        "seed": 1234,
        "epochs": 10000,
        "learning_rate": 2e-4,
        "betas": [0.8, 0.99],
        "eps": 1e-9,
        "batch_size": 1,
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
        "ogg_to_wav":True,
        "build_csv": True,
        "data_sources": [ #Ensure all datasets are in zip files
            #("gdrive", ),
            #("bucket", "speech_collection_bucket" ,"VALIDATED/acholi-validated.zip")
            #("bucket", "speech_collection_bucket" ,"VALIDATED/lugbara-validated.zip")
            ("bucket", "speech_collection_bucket" ,"VALIDATED/luganda-validated.zip")
            #("bucket", "speech_collection_bucket" ,"VALIDATED/runyankole-validated.zip")
            #("bucket", "speech_collection_bucket" ,"VALIDATED/ateso-validated.zip")
            #("bucket", "speech_collection_bucket" ,"VALIDATED/english-validated.zip")
        ],
        "bulild_csv": True,
        "language": "language",
        "lang_iso": "lug",
        "reference_file":"training_files/Prompt-Luganda.csv",
        "training_files":"training_files/acholi_multi_train.csv",
        "validation_files":"training_files/acholi_multi_val_n_test.csv",
        "data_root_dir": "dataset",
        "dataset_dir": "dataset/luganda-validated", 
        "text_cleaners":["transliteration_cleaners"],
        "max_wav_value": 32768.0,
        "sampling_rate": 16000,
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mel_channels": 80,
        "mel_fmin": 0.0,
        "mel_fmax": None,
        "add_blank": True,
        "n_speakers": 109,
        "cleaned_text": True
    },
    "model": {
        "vocab_file": "lug/vocab.txt",
        "g_checkpoint_path": "/path/to/checkpoint",#
        "d_checkpoint_path": "/path/to/checkpoint",
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
