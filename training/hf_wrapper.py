from huggingface_hub import hf_hub_download

from .models import SynthesizerTrn
from .text.mappers import TextMapper
from .train_config import config as HARDCODED_MODEL_CONFIG
from .utils import load_checkpoint

import torch


device = HARDCODED_MODEL_CONFIG["device"]

class VITSInfereceAdapterModel:

    def __init__(self, model_path, config_path, vocab_path, repo_name):
        self.repo_name = repo_name
        #self.hps = self._download_and_load_config(config_path)
        self.text_mapper = self._download_and_load_vocab(vocab_path)
        self.net_g  = SynthesizerTrn(
                len(self.text_mapper.symbols),
                HARDCODED_MODEL_CONFIG["data"]["filter_length"] // 2 + 1,
                HARDCODED_MODEL_CONFIG["train"]["segment_size"] // HARDCODED_MODEL_CONFIG["data"]["hop_length"],
                **HARDCODED_MODEL_CONFIG['model'])
        self.net_g = self._download_and_load_model(model_path, self.net_g, None)

    def _download_and_load_model(self, model_path, model, optimizer):
        # Use hf_hub_download to download the model
        model_file = hf_hub_download(repo_id=self.repo_name, filename=model_path)
        # Load the model using your custom logic
        load_checkpoint(model_file, model, optimizer)

        return model

    #def _download_and_load_config(self, config_path):
    #    # Similar logic for downloading and loading the config
    #    config_file = hf_hub_download(repo_id=self.repo_name, filename=model_path)

    def _download_and_load_vocab(self, vocab_path):
        # Similar logic for downloading and loading the vocab
        vocab_file = hf_hub_download(repo_id=self.repo_name, filename=vocab_path)
        text_mapper = TextMapper(vocab_file)
        return text_mapper

    def encode_text(self, txt):
        #txt = preprocess_text(txt, self.text_mapper, lang=LANG)
        stn_tst = self.text_mapper.get_text(txt)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
            hyp = self.model.infer(
                x_tst, x_tst_lengths,0, noise_scale=.667,
                noise_scale_w=0.8, length_scale=1.0
            )[0][0].detach().cpu()
        return hyp


    @classmethod
    def from_pretrained(cls, repo_name, G_net_path = "G_eng_lug.pth", vocab_path="vocab.txt", config_path = None  ):
        # Logic to instantiate the model using the repository name
        #G_net_path = "path_to_model_within_repo"
        return cls(G_net_path, config_path, vocab_path, repo_name)
