import os
import glob
import torch
import json
import logging
import subprocess
import unittest
import numpy as np

from training.utils import *

class MockLogger:
        def info(self, msg):
            print("[INFO]:", msg)
        def warn(self, msg):
            print("[WARN]:", msg)

logger = MockLogger()

class MockArgs:
        def __init__(self, model, config="./configs/base.json"):
            self.model = model
            self.config = config
            
def argparse_parser():
        return MockArgs(model="test_model")

argparse.ArgumentParser = argparse_parser

class MockWriter:
        def add_scalar(self, k, v, global_step):
            pass
        def add_histogram(self, k, v, global_step):
            pass
        def add_image(self, k, v, global_step, dataformats):
            pass
        def add_audio(self, k, v, global_step, sampling_rate):
            pass

def glob_mock(path):
    if "G_*.pth" in path:
        # Return three dummy paths for testing purposes
        return ['./dir_path/G_1.pth', './dir_path/G_2.pth', './dir_path/G_3.pth']
    return []

glob.glob = glob_mock

class TestUtils(unittest.TestCase):

    # Test for load_checkpoint
    def test_load_checkpoint():
        checkpoint_path = "./test_checkpoint.pth"
        dummy_model = torch.nn.Linear(5, 2)
        dummy_optimizer = torch.optim.Adam(dummy_model.parameters())
        
        save_checkpoint(dummy_model, dummy_optimizer, 0.001, 1, checkpoint_path)
        
        loaded_model, loaded_optimizer, lr, iteration = load_checkpoint(checkpoint_path, dummy_model, dummy_optimizer)
        
        assert lr == 0.001
        assert iteration == 1
        os.remove(checkpoint_path)


    # Test for save_checkpoint
    def test_save_checkpoint():
        dummy_model = torch.nn.Linear(5, 2)
        dummy_optimizer = torch.optim.Adam(dummy_model.parameters())
        checkpoint_path = "./test_checkpoint.pth"
        
        save_checkpoint(dummy_model, dummy_optimizer, 0.001, 1, checkpoint_path)
        assert os.path.exists(checkpoint_path)
        os.remove(checkpoint_path)



    # Test for latest_checkpoint_path
    def test_latest_checkpoint_path():
        dummy_paths = ['./dir_path/G_1.pth', './dir_path/G_2.pth', './dir_path/G_3.pth']
        for path in dummy_paths:
            with open(path, 'w') as f:
                f.write("dummy content")

        latest_path = latest_checkpoint_path('./dir_path')
        assert latest_path == './dir_path/G_3.pth'

        for path in dummy_paths:
            os.remove(path)

    # ... [Your previous mock and tests]

    # Mocking more functionalities for testing purposes
    

    # You would need to add more mocks if other external dependencies are called during the tests.

    # Test for summarize
    def test_summarize():
        writer = MockWriter()
        scalars = {"loss": 0.1}
        histograms = {"hist": [0.1, 0.2, 0.3]}
        images = {"img": np.array([[1,2],[3,4]])}
        audios = {"audio": np.array([0.1, 0.2, 0.3])}

        # Just testing if it runs without error, as there's no return value or side effect
        summarize(writer, 1, scalars=scalars, histograms=histograms, images=images, audios=audios)

    test_summarize()


    # Test for get_hparams
    def test_get_hparams():
        # Mocking os functionalities
        os.makedirs = lambda x: None
        with open('./configs/base.json', 'w') as f:
            f.write('{"param1": 10, "param2": "test"}')
        hparams = get_hparams(init=True)
        assert hparams.param1 == 10
        assert hparams.param2 == "test"
        os.remove('./configs/base.json')

    test_get_hparams()


    # Test for get_hparams_from_dir
    def test_get_hparams_from_dir():
        model_dir = './logs/test_model'
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            f.write('{"param1": 20, "param2": "test_dir"}')
        hparams = get_hparams_from_dir(model_dir)
        assert hparams.param1 == 20
        assert hparams.param2 == "test_dir"
        os.remove(os.path.join(model_dir, 'config.json'))



    # Test for get_hparams_from_file
    def test_get_hparams_from_file():
        config_path = './test_config.json'
        with open(config_path, 'w') as f:
            f.write('{"param1": 30, "param2": "test_file"}')
        hparams = get_hparams_from_file(config_path)
        assert hparams.param1 == 30
        assert hparams.param2 == "test_file"
        os.remove(config_path)


    import argparse

    # ... [Your previous code and tests]

    # Mocking more functionalities for testing purposes
    os.path.join = lambda *args: "/".join(args)

    # Test for load_checkpoint
    def test_load_checkpoint():
        # Mocking
        checkpoint_path = 'path_to_checkpoint'
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters())
        os.path.isfile = lambda x: True
        torch.load = lambda x, map_location: {
            'iteration': 1,
            'learning_rate': 0.001,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict()
        }
        # Test function
        m, o, lr, it = load_checkpoint(checkpoint_path, model, optimizer)
        assert it == 1
        assert lr == 0.001



    # Test for save_checkpoint
    def test_save_checkpoint():
        # Mocking
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters())
        torch.save = lambda *args, **kwargs: None
        checkpoint_path = 'path_to_checkpoint'
        # Test function
        save_checkpoint(model, optimizer, 0.001, 1, checkpoint_path)


    # Test for latest_checkpoint_path
    def test_latest_checkpoint_path():
        # Mocking
        dir_path = './dir_path'
        x = latest_checkpoint_path(dir_path)
        assert x == './dir_path/G_3.pth'

    # Test for check_git_hash
    def test_check_git_hash():
        # Mocking
        os.path.exists = lambda x: True
        subprocess.getoutput = lambda cmd: "test_hash"
        model_dir = './logs/test_model'
        with open(os.path.join(model_dir, 'githash'), 'w') as f:
            f.write('test_hash')
        # Test function
        check_git_hash(model_dir)
        with open(os.path.join(model_dir, 'githash'), 'r') as f:
            assert f.read() == 'test_hash'
        os.remove(os.path.join(model_dir, 'githash'))

if __name__ == "__main__":
    unittest.main()