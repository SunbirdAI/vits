from huggingface_hub import hf_hub_download

from .models import SynthesizerTrn
from .text.mappers import TextMapper
from .train_config import config as HARDCODED_MODEL_CONFIG
from .utils import load_checkpoint

import os
import torch


device = HARDCODED_MODEL_CONFIG["device"]

class VITSInfereceAdapterModel:

    """
    Adapter class for performing inference using the VITS model.
    
    Attributes:
        repo_name (str): The repository name on HuggingFace where the model and related files are hosted.
        hps: Model hyperparameters, loaded from the config file.
        text_mapper: Utility for mapping text to tokens, loaded from the vocabulary file.
        model: The VITS model instance.
        net_g: Loaded model from the provided checkpoint.
    """

    def __init__(self, model_path, config_path, vocab_path, repo_name):
        """
        Initialize the VITSInfereceAdapterModel.
        
        Args:
            model_path (str): Path to the model checkpoint file.
            config_path (str): Path to the configuration file.
            vocab_path (str): Path to the vocabulary file.
            repo_name (str): HuggingFace repository name where the model and related files are hosted.
        """

        self.repo_name = repo_name
        #self.hps = self._download_and_load_config(config_path)
        self.text_mapper = self._download_and_load_vocab(vocab_path)
        self.model  = SynthesizerTrn(
                len(self.text_mapper.symbols),
                HARDCODED_MODEL_CONFIG["data"]["filter_length"] // 2 + 1,
                HARDCODED_MODEL_CONFIG["train"]["segment_size"] // HARDCODED_MODEL_CONFIG["data"]["hop_length"],
                **HARDCODED_MODEL_CONFIG['model'])
        self.model = self._download_and_load_model(model_path)

    def _download_and_load_model(self, model_path):
        """
        Download the model from HuggingFace and load it.
        
        Args:
            model_path (str): Path to the model checkpoint file.
            model: The model instance to load the checkpoint into.
            optimizer: The optimizer associated with the model.
            
        Returns:
            model: The loaded model instance.
        """
        
        # Check if the model_path is a local directory and the file exists
        if os.path.exists(model_path):
            model_file = model_path
        else:
            # Attempt to download from HuggingFace
            model_file = hf_hub_download(repo_id=self.repo_name, filename=model_path)

        # Load the model using your custom logic
        load_checkpoint(model_file, self.model, None)

        return self.model

    #def _download_and_load_config(self, config_path):
    #    # Similar logic for downloading and loading the config
    #    config_file = hf_hub_download(repo_id=self.repo_name, filename=model_path)

    
    def _download_and_load_vocab(self, vocab_path):
        """
        Download the vocabulary from HuggingFace and load it.
        
        Args:
            vocab_path (str): Path to the vocabulary file.
            
        Returns:
            TextMapper: Instance of the TextMapper utility loaded with the vocabulary.
        """
        
        # Check if the vocab_path is a local directory and the file exists
        if os.path.exists(vocab_path):
            vocab_file = vocab_path
        else:
            # Attempt to download from HuggingFace
            vocab_file = hf_hub_download(repo_id=self.repo_name, filename=vocab_path)

        text_mapper = TextMapper(vocab_file)
        return text_mapper

    def encode_text(self, txt, cleaner_names, cleaner_regex = None):
        """
        Encode a given text using the VITS model.
        
        Args:
            txt (str): Text to be encoded.
            
        Returns:
            Tensor: Encoded representation of the input text.
        """
        #txt = preprocess_text(txt, self.text_mapper, lang=LANG)
        stn_tst = self.text_mapper.get_text(txt, cleaner_names, cleaner_regex )
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
            hyp = self.model.infer(
                x_tst, x_tst_lengths,0, noise_scale=.667,
                noise_scale_w=0.8, length_scale=1.0
            )[0][0].detach().cpu()
        return hyp

    #FIXME remove it?
    @classmethod
    def from_pretrained(cls, repo_name, G_net_path = "G_eng_lug.pth", vocab_path="vocab.txt", config_path = None  ):
        """
        Class method to instantiate an instance of VITSInfereceAdapterModel from a pre-trained checkpoint.
        
        Args:
            repo_name (str): HuggingFace repository name where the model and related files are hosted.
            G_net_path (str, optional): Path to the model checkpoint within the repository. Defaults to "G_eng_lug.pth".
            vocab_path (str, optional): Path to the vocabulary file within the repository. Defaults to "vocab.txt".
            config_path (str, optional): Path to the configuration file within the repository.
            
        Returns:
            VITSInfereceAdapterModel: An instance of the VITSInfereceAdapterModel.
        """
        # Logic to instantiate the model using the repository name
        #G_net_path = "path_to_model_within_repo"
        return cls(G_net_path, config_path, vocab_path, repo_name)
