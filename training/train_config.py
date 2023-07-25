config = {
    #"training_csv": "acholi_solomon_train.csv",
    #"val_csv": "acholi_solomon_val.csv",
    #"test_csv": "acholi_solomon_test.csv",
    "model_dir": "best",
    "multispeaker": True,
    "ckpt_dir": None,
    "device": "cpu",
    "vertex": {
        "gcp_project": "sb-gcp-project-01",
        "bucket_name": "ali_speech_experiments",
        "gcp_region": "europe-west6",
        "app_name": "train_tts", #according to //// format
        "prebuilt_docker_image": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-7:latest",
        "package_application_dir":"training",
        "source_package_file_name": "{}/dist/trainer-0.1.tar.gz", #root_dir same as package_application_dir
        "python_package_gcs_uri": "{}/pytorch-on-gcp/{}/train/python_package/trainer-0.1.tar.gz", #bucket_name app_name
        "python_module_name": "training.run" #To run?
        "requirements": [
            "Cython==0.29.21",
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
        "batch_size": 64,
        "fp16_run": True,
        "lr_decay": 0.999875,
        "segment_size": 8192,
        "init_lr_ratio": 1,
        "warmup_epochs": 0,
        "c_mel": 45,
        "c_kl": 1.0
    },
    "data": {
        "training_files":"training_files/acholi_train_mono.csv",
        "validation_files":"training_files/acholi_val_n_test_mono.csv",
        "data_root_dir": "/media/ali/Vault 1/test/vits/",
        "text_cleaners":["transliteration_cleaners"],
        "max_wav_value": 32768.0,
        "sampling_rate": 24000,
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
        "vocab_file": "/media/ali/Vault 1/test/vits/ach/vocab.txt",
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
