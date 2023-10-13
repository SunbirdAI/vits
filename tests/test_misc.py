import os
import pandas as pd
import shutil
import tempfile
import unittest
from unittest.mock import patch, mock_open, MagicMock

import librosa
import soundfile as sf
import re
import zipfile
from tqdm import tqdm

from training.misc import *

class TestDownloadFunction(unittest.TestCase):

    def setUp(self):
        # Temporary directory for testing
        self.test_dir = "test_download_dir"
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)

    def tearDown(self):
        # Cleanup the directory after tests
        shutil.rmtree(self.test_dir)

    @patch("subprocess.check_output")
    def test_download_when_dir_exists(self, mock_subprocess):
        # If the directory already exists, the function should return the directory
        lang = "test_lang"
        os.mkdir(os.path.join(self.test_dir, lang))

        result = download(lang, self.test_dir)
        self.assertEqual(result, os.path.join(self.test_dir, lang))
        mock_subprocess.assert_not_called()  # Ensure the download is skipped

    @patch("subprocess.check_output")
    def test_download_when_dir_does_not_exist(self, mock_subprocess):
        # If the directory does not exist, it should call subprocess for download
        lang = "test_lang2"

        result = download(lang, self.test_dir)
        self.assertEqual(result, lang)  # This is how the function currently behaves
        mock_subprocess.assert_called_once()

    def test_non_existent_target_directory(self):
        # If the target directory does not exist, it should be created
        lang = "test_lang3"
        new_tgt_dir = os.path.join(self.test_dir, "new_dir")
        self.assertFalse(os.path.exists(new_tgt_dir))

        download(lang, new_tgt_dir)
        self.assertTrue(os.path.exists(new_tgt_dir))


class TestDownloadFunction(unittest.TestCase):

    def setUp(self):
        # Temporary directory for testing
        self.test_dir = "test_download_dir"
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)

    def tearDown(self):
        # Cleanup the directory after tests
        shutil.rmtree(self.test_dir)

    @patch("subprocess.check_output")
    def test_download_when_dir_exists(self, mock_subprocess):
        # If the directory already exists, the function should return the directory
        lang = "test_lang"
        os.mkdir(os.path.join(self.test_dir, lang))

        result = download(lang, self.test_dir)
        self.assertEqual(result, os.path.join(self.test_dir, lang))
        mock_subprocess.assert_not_called()  # Ensure the download is skipped

    @patch("subprocess.check_output")
    def test_download_when_dir_does_not_exist(self, mock_subprocess):
        # If the directory does not exist, it should call subprocess for download
        lang = "test_lang2"

        result = download(lang, self.test_dir)
        self.assertEqual(result, lang)  # This is how the function currently behaves
        mock_subprocess.assert_called_once()

    def test_non_existent_target_directory(self):
        # If the target directory does not exist, it should be created
        lang = "test_lang3"
        new_tgt_dir = os.path.join(self.test_dir, "new_dir")
        self.assertFalse(os.path.exists(new_tgt_dir))

        download(lang, new_tgt_dir)
        self.assertTrue(os.path.exists(new_tgt_dir))

class TestDownloaderFunctions(unittest.TestCase):

    def setUp(self):
        pass

    @patch('os.path.exists', return_value=False)
    @patch('os.mkdir')
    def test_download_blob_creates_directory(self, mock_mkdir, mock_exists):
        with patch.object(storage.Client, 'from_service_account_json') as mock_gcp:
            mock_bucket = MagicMock()
            mock_gcp.return_value.bucket.return_value = mock_bucket
            mock_bucket.list_blobs.return_value = []
            download_blob("mock_bucket", "mock_source", "mock_destination")
            mock_mkdir.assert_called_once()

    @patch('os.path.exists', return_value=True)
    @patch.object(storage.Blob, 'download_to_filename')
    def test_download_blob_downloads_file(self, mock_download, mock_exists):
        with patch.object(storage.Client, 'from_service_account_json') as mock_gcp:
            mock_bucket = MagicMock()
            mock_gcp.return_value.bucket.return_value = mock_bucket

            mock_blob = MagicMock(spec=storage.Blob)
            mock_bucket.list_blobs.return_value = [mock_blob]

            download_blob("mock_bucket", "mock_source", "mock_destination")
            mock_download.assert_called_once()

    @patch('zipfile.ZipFile.extractall')
    def test_download_and_extract_zip_file(self, mock_extractall):
        with patch('gdown.download', return_value='dummy_path.zip'):
            download_and_extract_drive_file('mock_file_id', 'mock_destination_folder')
            mock_extractall.assert_called_once()

    @patch('os.remove')
    def test_zip_file_removal_after_extraction(self, mock_remove):
        with patch('gdown.download', return_value='dummy_path.zip'), patch('zipfile.ZipFile'):
            download_and_extract_drive_file('mock_file_id', 'mock_destination_folder')
            mock_remove.assert_called_once()

    @patch('os.path.exists', return_value=True)
    def test_download_blob_with_no_blobs(self, mock_exists):
        with patch.object(storage.Client, 'from_service_account_json') as mock_gcp:
            mock_bucket = MagicMock()
            mock_print = MagicMock()
            mock_gcp.return_value.bucket.return_value = mock_bucket

            mock_bucket.list_blobs.return_value = []
            download_blob("mock_bucket", "mock_source", "mock_destination")

            # Assert that the print function was called with the right argument
            with patch("builtins.print", mock_print):
                download_blob("mock_bucket", "nonexistent_source", "mock_destination")
                mock_print.assert_called_with("Blob nonexistent_source downloaded to mock_destination.")

    @patch('gdown.download', return_value='dummy_path.txt')
    @patch('zipfile.ZipFile')
    def test_download_non_zip_file(self, mock_zip, mock_download):
        # Should not raise an exception or error
        download_and_extract_drive_file('mock_file_id', 'mock_destination_folder')
        mock_zip.assert_not_called()

    def test_exception_handling_on_corrupt_zip(self):
        with patch('gdown.download', return_value='dummy_path.zip'):
            with patch('zipfile.ZipFile.extractall', side_effect=zipfile.BadZipFile):
                # Asserting exception is raised
                with self.assertRaises(zipfile.BadZipFile):
                    download_and_extract_drive_file('mock_file_id', 'mock_destination_folder')


class TestDataCreationFunctions(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_csv = os.path.join(self.temp_dir, 'temp.csv')

    @patch('os.walk')
    @patch('pd.read_csv')
    def test_create_multispeaker_audio_csv(self, mock_read_csv, mock_os_walk):
        mock_read_csv.return_value = pd.DataFrame({'Key': [1], 'Text': ['Sample']})
        mock_os_walk.return_value = [(self.temp_dir, [], ['sample.wav'])]

        create_multispeaker_audio_csv(self.temp_dir, self.temp_csv, train_csv='train.csv', val_test_csv='val_test.csv')
        
        self.assertTrue(os.path.exists('train.csv'))
        self.assertTrue(os.path.exists('val_test.csv'))

        os.remove('train.csv')
        os.remove('val_test.csv')

    @patch('os.walk')
    @patch('pd.read_csv')
    def test_create_multilingual_audio_csv(self, mock_read_csv, mock_os_walk):
        mock_read_csv.return_value = pd.DataFrame({'Key': [1], 'Text': ['Sample']})
        mock_os_walk.return_value = [(self.temp_dir, [], ['sample.wav'])]
        
        create_multilingual_audio_csv([self.temp_dir], [self.temp_csv], train_csv='train_multilang.csv', val_test_csv='val_test_multilang.csv')
        
        self.assertTrue(os.path.exists('train_multilang.csv'))
        self.assertTrue(os.path.exists('val_test_multilang.csv'))

        os.remove('train_multilang.csv')
        os.remove('val_test_multilang.csv')

    @patch('os.walk')
    @patch('pd.read_csv')
    def test_invalid_file_in_directory(self, mock_read_csv, mock_os_walk):
        mock_read_csv.return_value = pd.DataFrame({'Key': [1], 'Text': ['Sample']})
        mock_os_walk.return_value = [(self.temp_dir, [], ['sample.txt'])]  # this is a non-wav file

        # Calling the function should not raise an exception even with invalid file types
        create_multispeaker_audio_csv(self.temp_dir, self.temp_csv, train_csv='train.csv', val_test_csv='val_test.csv')

        self.assertTrue(os.path.exists('train.csv'))
        self.assertTrue(os.path.exists('val_test.csv'))
        
        os.remove('train.csv')
        os.remove('val_test.csv')

    def test_missing_key_in_csv(self):
        with self.assertRaises(Exception):
            # Let's consider a directory name that doesn't match any key in the CSV
            create_multispeaker_audio_csv(self.temp_dir, "non_existent.csv")

    @patch('os.walk')
    def test_empty_directory(self, mock_os_walk):
        mock_os_walk.return_value = [(self.temp_dir, [], [])]  # empty directory
        # An empty directory should not cause an exception
        create_multispeaker_audio_csv(self.temp_dir, self.temp_csv)

    @patch('os.walk')
    @patch('pd.read_csv')
    def test_noninteger_keys(self, mock_read_csv, mock_os_walk):
        mock_read_csv.return_value = pd.DataFrame({'Key': ['non-integer'], 'Text': ['Sample']})
        mock_os_walk.return_value = [(self.temp_dir, [], ['sample.wav'])]
        # Non-integer keys should not cause an exception
        create_multispeaker_audio_csv(self.temp_dir, self.temp_csv)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

class TestFunctions(unittest.TestCase):

    def test_convert_and_resample(self):
        directory = '/mock/directory/path'

        with patch('os.walk') as mock_os_walk:
            mock_os_walk.return_value = [(directory, [], ['test.ogg'])]
            with patch('AudioSegment.from_ogg') as mock_from_ogg:
                audio_mock = MagicMock()
                mock_from_ogg.return_value = audio_mock
                convert_and_resample('fake_dir', 16000)
                mock_from_ogg.assert_called()
                audio_mock.export.assert_called()

    def test_find_non_allowed_characters_multispeaker(self):
        mock_data = {'path': ['path1', 'path2'], 'speaker': [1, 2], 'transcription': ['hello', 'world']}
        df = pd.DataFrame(mock_data)
        vocab = ['h', 'e', 'l', 'o', 'w', 'r', 'd']
        df.to_csv('temp.csv', sep='|', index=False, header=False)
        result = find_non_allowed_characters(['temp.csv'], vocab, multispeaker=True)
        self.assertEqual(result, set())

    def test_find_non_allowed_characters_single(self):
        mock_data = {'path': ['path1', 'path2'], 'transcription': ['hello', 'world!']}
        df = pd.DataFrame(mock_data)
        vocab = ['h', 'e', 'l', 'o', 'w', 'r', 'd']
        df.to_csv('temp.csv', sep='|', index=False, header=False)
        result = find_non_allowed_characters(['temp.csv'], vocab, multispeaker=False)
        self.assertEqual(result, {'!'})

    def test_create_regex_for_character_list(self):
        char_list = ['a', 'b', '*', '+']
        pattern = create_regex_for_character_list(char_list)
        self.assertEqual(pattern, r'a|b|\*|\+')

    def test_check_nan(self):
        tensor = torch.Tensor([1, 2, 3])
        with self.assertRaises(ValueError):
            check_nan(tensor, 'tensor', logger=None)

if __name__ == '__main__':
    unittest.main()




