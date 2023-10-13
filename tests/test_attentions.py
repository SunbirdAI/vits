import torch
import torch.nn.functional as F
import unittest

from training.attentions import *

class TestModels(unittest.TestCase):

    def test_encoder(self):
        # Synthetic input
        batch_size, channels, timesteps = 32, 64, 16
        x = torch.rand(batch_size, channels, timesteps)
        x_mask = torch.ones(batch_size, 1, timesteps)
        
        # Initialize the model
        model = Encoder(hidden_channels=channels, filter_channels=128, n_heads=4, n_layers=2)
        out = model(x, x_mask)
        
        # Check output shape
        self.assertEqual(out.shape, (batch_size, channels, timesteps))

    def test_decoder(self):
        # Synthetic input
        batch_size, channels, timesteps = 32, 64, 16
        x = torch.rand(batch_size, channels, timesteps)
        h = torch.rand(batch_size, channels, timesteps)
        x_mask = torch.ones(batch_size, 1, timesteps)
        h_mask = torch.ones(batch_size, 1, timesteps)
        
        # Initialize the model
        model = Decoder(hidden_channels=channels, filter_channels=128, n_heads=4, n_layers=2)
        out = model(x, x_mask, h, h_mask)
        
        # Check output shape
        self.assertEqual(out.shape, (batch_size, channels, timesteps))

    def test_multihead_attention(self):
        # Synthetic input
        batch_size, channels, timesteps = 32, 64, 16
        x = torch.rand(batch_size, channels, timesteps)
        c = torch.rand(batch_size, channels, timesteps)
        
        # Initialize the model
        model = MultiHeadAttention(channels, channels, n_heads=4)
        out = model(x, c)
        
        # Check output shape
        self.assertEqual(out.shape, (batch_size, channels, timesteps))

    def test_ffn(self):
        # Synthetic input
        batch_size, channels, timesteps = 32, 64, 16
        x = torch.rand(batch_size, channels, timesteps)
        x_mask = torch.ones(batch_size, 1, timesteps)
        
        # Initialize the model
        model = FFN(in_channels=channels, out_channels=channels, filter_channels=128, kernel_size=3)
        out = model(x, x_mask)
        
        # Check output shape
        self.assertEqual(out.shape, (batch_size, channels, timesteps))


if __name__ == '__main__':
    unittest.main()
