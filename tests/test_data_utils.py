import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import torch
import random


# Assume the TextAudioLoader class is in a file named 'audio_utils.py'
from ..training.data_utils import *

class TestAudioFunctions(unittest.TestCase):
    
    def test_check_audio_file_valid(self):
        # Mock torchaudio.load to simulate a valid file
        with patch('torchaudio.load', return_value=(None, None)):
            result = check_audio_file("valid_file.wav")
            self.assertTrue(result["is_audio_ok"])
            self.assertEqual(result["file_path"], "valid_file.wav")
            
    def test_check_audio_file_invalid(self):
        # Mock torchaudio.load to raise an exception
        with patch('torchaudio.load', side_effect=Exception("Error")):
            result = check_audio_file("invalid_file.wav")
            self.assertFalse(result["is_audio_ok"])
            self.assertEqual(result["file_path"], "invalid_file.wav")
            
    def test_verify_audio_dir(self):
        fake_files = ["valid1.wav", "valid2.wav", "invalid.mp3"]
        results = [{"file_path": "path/valid1.wav", "is_audio_ok": True},
                   {"file_path": "path/valid2.wav", "is_audio_ok": True}]
        
        # Mock os.walk to simulate directory contents and torchaudio.load for audio check
        with patch('os.walk', return_value=[("path", None, fake_files)]):
            with patch('torchaudio.load', return_value=(None, None)):
                df = verify_audio_dir("audio_directory")
                pd.testing.assert_frame_equal(df, pd.DataFrame(results))
                
    # More tests can be added, e.g., for different file extensions, nested directories, etc.

class TestTextAudioLoader(unittest.TestCase):

    def setUp(self):
        self.hparams = {
            "data_root_dir": "/path/to/data",
            "text_cleaners": [],
            "max_wav_value": 1.0,
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "add_blank": False,
            "min_text_len": 1,
            "max_text_len": 190
        }
        
        # Mock text_mapper
        self.text_mapper = lambda: None
        self.text_mapper.text_to_sequence = lambda x, y: [1, 2, 3]

    @patch("audio_utils.load_wav_to_torch")
    @patch("os.path.getsize", return_value=160000)
    @patch("os.path.exists", return_value=False)
    @patch("torch.load")
    @patch("torch.save")
    def test_valid_audio_text_pair(self, mock_torch_save, mock_torch_load, mock_os_path_exists, mock_os_path_getsize, mock_load_wav_to_torch):
        mock_load_wav_to_torch.return_value = (torch.rand((22050,)), 22050)
        
        dataset = TextAudioLoader(["file1.wav|hello", "file2.wav|world"], self.hparams, self.text_mapper)

        # Test __getitem__
        text, spec, wav = dataset[0]
        self.assertTrue(isinstance(text, torch.LongTensor))
        self.assertTrue(isinstance(spec, torch.Tensor))
        self.assertTrue(isinstance(wav, torch.Tensor))
        self.assertTrue(wav.dim() == 2)
        self.assertTrue(spec.dim() == 2)

        # Test __len__
        self.assertEqual(len(dataset), 2)

    @patch("audio_utils.load_wav_to_torch", side_effect=Exception("Invalid audio"))
    @patch("os.path.getsize", return_value=160000)
    def test_invalid_audio(self, mock_os_path_getsize, mock_load_wav_to_torch):
        with self.assertRaises(ValueError):
            dataset = TextAudioLoader(["invalid_file.wav|hello"], self.hparams, self.text_mapper)
            dataset[0]

    @patch("audio_utils.load_wav_to_torch", return_value=(torch.rand((22050,)), 44100))
    @patch("os.path.getsize", return_value=160000)
    def test_different_sampling_rate(self, mock_os_path_getsize, mock_load_wav_to_torch):
        with self.assertRaises(ValueError):
            dataset = TextAudioLoader(["diff_sampling_file.wav|hello"], self.hparams, self.text_mapper)
            dataset[0]

    @patch("audio_utils.load_wav_to_torch", return_value=(torch.rand((22050,)), 22050))
    @patch("os.path.getsize", return_value=160000)
    def test_text_length_out_of_bounds(self, mock_os_path_getsize, mock_load_wav_to_torch):
        # Too short
        dataset = TextAudioLoader(["file_short.wav|"], self.hparams, self.text_mapper)
        self.assertEqual(len(dataset), 0)

        # Too long
        long_text = "a" * 200
        dataset = TextAudioLoader([f"file_long.wav|{long_text}"], self.hparams, self.text_mapper)
        self.assertEqual(len(dataset), 0)

    @patch("audio_utils.load_wav_to_torch", return_value=(torch.rand((22050,)), 22050))
    @patch("os.path.getsize", return_value=160000)
    @patch("os.path.exists", return_value=True)
    def test_spectrogram_exists(self, mock_os_path_exists, mock_os_path_getsize, mock_load_wav_to_torch):
        # Mocking torch.load to simulate the existence of spectrogram file.
        mock_spec = torch.rand((80, 100))
        with patch("torch.load", return_value=mock_spec):
            dataset = TextAudioLoader(["existing_spec_file.wav|hello"], self.hparams, self.text_mapper)
            _, spec, _ = dataset[0]
            self.assertTrue(torch.equal(spec, mock_spec))


class MockDataset:
    def __init__(self, lengths):
        self.lengths = lengths

class TestDistributedBucketSampler(unittest.TestCase):

    def setUp(self):
        # Example lengths for a mock dataset
        self.lengths = [10, 15, 20, 25, 30, 45, 50, 60, 75, 80, 90, 100]
        self.dataset = MockDataset(self.lengths)

    def test_basic_functionality(self):
        sampler = DistributedBucketSampler(self.dataset, batch_size=2, boundaries=[0, 30, 60, 100], num_replicas=2, rank=0, shuffle=False)

        batches = list(sampler)

        # Check bucketing
        for batch in batches:
            lengths_in_batch = [self.lengths[i] for i in batch]
            min_len, max_len = min(lengths_in_batch), max(lengths_in_batch)
            self.assertTrue(any(bound <= min_len <= bound_next and bound < max_len <= bound_next for bound, bound_next in zip([0, 30, 60], [30, 60, 100])))

    def test_distributed_buckets(self):
        sampler0 = DistributedBucketSampler(self.dataset, batch_size=2, boundaries=[0, 30, 60, 100], num_replicas=2, rank=0, shuffle=False)
        sampler1 = DistributedBucketSampler(self.dataset, batch_size=2, boundaries=[0, 30, 60, 100], num_replicas=2, rank=1, shuffle=False)

        batches0 = list(sampler0)
        batches1 = list(sampler1)

        # The two samplers should not have common batches
        self.assertTrue(all(batch0 != batch1 for batch0 in batches0 for batch1 in batches1))

    def test_total_size_and_length(self):
        sampler = DistributedBucketSampler(self.dataset, batch_size=2, boundaries=[0, 30, 60, 100], num_replicas=2, rank=0, shuffle=False)

        self.assertEqual(len(sampler), sampler.total_size // (2 * sampler.batch_size))

    def test_bucket_boundaries(self):
        sampler = DistributedBucketSampler(self.dataset, batch_size=2, boundaries=[0, 30, 60, 100], num_replicas=2, rank=0, shuffle=False)

        self.assertEqual(sampler._bisect(5), -1)    # Outside the boundaries
        self.assertEqual(sampler._bisect(40), 1)   # Between 30 and 60
        self.assertEqual(sampler._bisect(95), 2)   # Between 60 and 100
        self.assertEqual(sampler._bisect(110), -1) # Outside the boundaries



if __name__ == '__main__':
    unittest.main()
