import torch
import pytest
from unittest.mock import Mock
from training.models import SynthesizerTrn

import torch
import pytest
from unittest.mock import Mock
from training.models import (StochasticDurationPredictor, DurationPredictor, TextEncoder, 
                                ResidualCouplingBlock, PosteriorEncoder, Generator, DiscriminatorP, 
                                DiscriminatorS, MultiPeriodDiscriminator)

import unittest


# Sample input data for the model's methods. This is just placeholder data; you might need to adjust it.
SAMPLE_INPUT = torch.randn(1, 100)
SAMPLE_INPUT_LENGTHS = torch.LongTensor([100])
SAMPLE_SID = torch.LongTensor([0])
SAMPLE_Y = torch.randn(1, 22050)
SAMPLE_Y_LENGTHS = torch.LongTensor([22050])

class TestMe(unittest.TestCase):
    @pytest.fixture(scope="training")
    def synthesizer_model():
        # Instantiate the model with mock parameters
        model = SynthesizerTrn(
            n_vocab=20,
            spec_channels=80,
            segment_size=1024,
            inter_channels=512,
            hidden_channels=128,
            filter_channels=256,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.1,
            resblock="MockResblock", 
            resblock_kernel_sizes=[3, 7], 
            resblock_dilation_sizes=[1, 3], 
            upsample_rates=[2, 4],
            upsample_initial_channel=128, 
            upsample_kernel_sizes=[3, 7],
            n_speakers=10,
            gin_channels=128,
            use_sdp=True
        )
        return model

    def test_forward(synthesizer_model):
        output = synthesizer_model(SAMPLE_INPUT, SAMPLE_INPUT_LENGTHS, SAMPLE_Y, SAMPLE_Y_LENGTHS, SAMPLE_SID)
        
        # You can add more detailed assertions based on the expected output shape and properties
        assert len(output) == 7, "Expected 7 output values"
        assert not torch.isnan(output[0]).any(), "Found NaNs in the output tensor"

    def test_infer(synthesizer_model):
        output = synthesizer_model.infer(SAMPLE_INPUT, SAMPLE_INPUT_LENGTHS, SAMPLE_SID)
        
        assert len(output) == 4, "Expected 4 output values"
        assert not torch.isnan(output[0]).any(), "Found NaNs in the output tensor"

    def test_voice_conversion(synthesizer_model):
        output = synthesizer_model.voice_conversion(SAMPLE_Y, SAMPLE_Y_LENGTHS, SAMPLE_SID, SAMPLE_SID)
        
        assert len(output) == 3, "Expected 3 output values"
        assert not torch.isnan(output[0]).any(), "Found NaNs in the output tensor"



    # Create a fixture for the device
    @pytest.fixture(scope="training")
    def device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def test_StochasticDurationPredictor(device):
        model = StochasticDurationPredictor(128, 64, 3, 0.1).to(device)
        x = torch.randn(8, 128, 32).to(device)
        x_mask = torch.ones(8, 32).to(device)
        output = model(x, x_mask)
        assert output.size(1) == 2, "Expected a different output size."
        assert not torch.isnan(output).any(), "Found NaNs in the output."


    def test_DurationPredictor(device):
        model = DurationPredictor(128, 64, 3, 0.1).to(device)
        x = torch.randn(8, 128, 32).to(device)
        x_mask = torch.ones(8, 32).to(device)
        output = model(x, x_mask)
        assert output.size(1) == 1, "Expected a different output size."
        assert not torch.isnan(output).any(), "Found NaNs in the output."


    def test_TextEncoder(device):
        model = TextEncoder(32, 64, 128, 64, 2, 3, 3, 0.1).to(device)
        x = torch.randint(0, 31, (8, 32)).to(device)
        x_lengths = torch.full((8,), 32).long().to(device)
        outputs = model(x, x_lengths)
        assert outputs[0].size(1) == 128, "Expected a different output size for the encoder output."
        assert not any(torch.isnan(o).any() for o in outputs), "Found NaNs in the outputs."


    def test_ResidualCouplingBlock(device):
        model = ResidualCouplingBlock(64, 128, 3, 1, 3).to(device)
        x = torch.randn(8, 64, 32).to(device)
        x_mask = torch.ones(8, 32).to(device)
        output = model(x, x_mask)
        assert output.size(1) == 64, "Expected a different output size."
        assert not torch.isnan(output).any(), "Found NaNs in the output."


    def test_PosteriorEncoder(device):
        model = PosteriorEncoder(32, 64, 128, 3, 1, 3).to(device)
        x = torch.randn(8, 32, 32).to(device)
        x_lengths = torch.full((8,), 32).long().to(device)
        outputs = model(x, x_lengths)
        assert outputs[0].size(1) == 64, "Expected a different output size for the encoder output."
        assert not any(torch.isnan(o).any() for o in outputs), "Found NaNs in the outputs."


    def test_Generator(device):
        model = Generator(32, '1', [3], [1], [2], 64, [3]).to(device)
        x = torch.randn(8, 32, 16).to(device)
        output = model(x)
        assert output.size(1) == 1, "Expected a different output size."
        assert not torch.isnan(output).any(), "Found NaNs in the output."


    def test_DiscriminatorP(device):
        model = DiscriminatorP(3).to(device)
        x = torch.randn(8, 1, 64, 32).to(device)
        outputs = model(x)
        assert outputs[0].size(1) == 1, "Expected a different output size for the discriminator output."
        assert not any(torch.isnan(o).any() for o in outputs), "Found NaNs in the outputs."


    def test_DiscriminatorS(device):
        model = DiscriminatorS().to(device)
        x = torch.randn(8, 1, 32).to(device)
        outputs = model(x)
        assert outputs[0].size(1) == 1, "Expected a different output size for the discriminator output."
    
if __name__ == "__main__":
    unittest.main()

# Run the tests:
# pytest tests/
