import unittest
import torch
import numpy as np
import librosa

from ..training.mel_processing import *

class TestAudioFunctions(unittest.TestCase):
    
    def setUp(self):
        # Random input for testing
        self.y = torch.rand(22050)  # 1 second worth of audio at 22050Hz
        self.n_fft = 2048
        self.num_mels = 80
        self.sampling_rate = 22050
        self.hop_size = 512
        self.win_size = 2048
        self.fmin = 0
        self.fmax = 8000
        self.logger = None

    def test_dynamic_range_compression_decompression(self):
        compressed = dynamic_range_compression_torch(self.y)
        decompressed = dynamic_range_decompression_torch(compressed)
        self.assertTrue(torch.isclose(self.y, decompressed, atol=1e-5).all())

    def test_spectral_normalize_denormalize(self):
        normalized = spectral_normalize_torch(self.y)
        denormalized = spectral_de_normalize_torch(normalized)
        self.assertTrue(torch.isclose(self.y, denormalized, atol=1e-5).all())

    def test_spectrogram_shape(self):
        spec = spectrogram_torch(self.y, self.n_fft, self.sampling_rate, self.hop_size, self.win_size)
        expected_shape = (self.n_fft // 2 + 1, -1)
        self.assertEqual(spec.shape, expected_shape)

    def test_spec_to_mel_shape(self):
        spec = spectrogram_torch(self.y, self.n_fft, self.sampling_rate, self.hop_size, self.win_size)
        mel = spec_to_mel_torch(spec, self.n_fft, self.num_mels, self.sampling_rate, self.fmin, self.fmax)
        expected_shape = (self.num_mels, spec.shape[1])
        self.assertEqual(mel.shape, expected_shape)

    def test_mel_spectrogram_shape(self):
        mel = mel_spectrogram_torch(self.y, self.n_fft, self.num_mels, self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax)
        expected_shape = (self.num_mels, -1)
        self.assertEqual(mel.shape, expected_shape)

    def test_mel_spectrogram_with_logger(self):
        mock_logger = MockLogger()
        mel = mel_spectrogram_torch(self.y, self.n_fft, self.num_mels, self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax, logger=mock_logger)
        self.assertFalse(mock_logger.has_logs)

class MockLogger:
    def __init__(self):
        self.logs = []

    def info(self, message):
        self.logs.append(message)

    @property
    def has_logs(self):
        return len(self.logs) > 0

if __name__ == '__main__':
    unittest.main()
