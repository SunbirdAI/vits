import pytest
from unittest.mock import patch, Mock
from training.hf_wrapper import VITSInfereceAdapterModel


raise NotImplementedError
# Mock the repo_name and model paths
REPO_NAME = "your-huggingface-repo-name"
MODEL_PATH = "mock_model.pth"
VOCAB_PATH = "mock_vocab.txt"
CONFIG_PATH = "mock_config.json"

# Mock the hf_hub_download to prevent actual download
@patch("your_module.models.hf_hub_download")
def mock_hf_hub_download(mocked_download, path):
    # Return the path as it is for testing purposes
    return path

@pytest.fixture(scope="module")
def model_instance(mock_hf_hub_download):
    return VITSInfereceAdapterModel.from_pretrained(REPO_NAME, MODEL_PATH, VOCAB_PATH, CONFIG_PATH)

def test_from_pretrained(model_instance):
    assert isinstance(model_instance, VITSInfereceAdapterModel)

    # You can also test specific attributes of the model, e.g.,
    assert model_instance.repo_name == REPO_NAME

def test_encode_text(model_instance):
    text = "Test text"
    result = model_instance.encode_text(text)
    assert result is not None
    
    # If you have expected output shapes/values, you can add checks
    # For example:
    # assert result.shape == (some_expected_shape)
