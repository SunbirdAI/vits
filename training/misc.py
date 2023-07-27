import gdown
from google.cloud import storage
import numpy as np
import os
import pandas as pd
import subprocess
import torchaudio
from tqdm import tqdm
from train_config import config
import zipfile

def download(lang, tgt_dir="./"):
  lang_fn, lang_dir = os.path.join(tgt_dir, lang+'.tar.gz'), os.path.join(tgt_dir, lang)
  cmd = ";".join([
        f"wget https://dl.fbaipublicfiles.com/mms/tts/full_model/{lang}.tar.gz -O {lang_fn}",
        f"tar zxvf {lang_fn}"
  ])
  #Check if this works ?
  if os.path.exists(lang_dir):
    return lang_dir

  subprocess.check_output(cmd, shell=True)
  print(f"Model checkpoints in {lang_dir}: {os.listdir(lang_dir)}")
  return lang_dir


def balance_speakers(csv_file_path, separator, use_median=False, prefix="balanced_"):
    # Read CSV into dataframe
    df = pd.read_csv(csv_file_path, sep=separator, header=None, names=["path", "speaker_id", "transcription"])

    # Group by 'speaker_id' and get minimum group size
    if use_median:
        group_size = df.groupby("speaker_id").size()
        balance_size = int(group_size.median())
    else:
        balance_size = df.groupby("speaker_id").size().min()

    # Create a list to hold the balanced dataframes
    balanced_dfs = []

    # For each 'speaker_id', randomly sample rows up to 'balance_size'
    for speaker_id, group_df in df.groupby("speaker_id"):
        balanced_df = group_df.sample(min(balance_size, len(group_df)))
        balanced_dfs.append(balanced_df)

    # Concatenate all balanced dataframes
    balanced_data = pd.concat(balanced_dfs)

    # Create new file name with prefix
    new_file_path = os.path.join(os.path.dirname(csv_file_path), prefix + os.path.basename(csv_file_path))

    # Write balanced dataframe to new CSV file
    balanced_data.to_csv(new_file_path, sep=separator, header=False, index=False)

    print(f"Balanced data written to: {new_file_path}")
    return new_file_path

def check_audio_file(batch):
    try:
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        return {"is_audio_ok": True}
    except Exception as e:
        print(f"Could not process file {batch['path']}. Error: {str(e)}")
        return {"is_audio_ok": False}

def filter_corrupt_files(csv_file_path, separator):
    # Read CSV into dataframe
    df = pd.read_csv(csv_file_path, sep=separator, header=None)

    # Create empty list to hold results
    results = []
    corrupt_files = 0

    # Iterate over the dataframe
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Check if audio file exists
        audio_path = row[0]
        audio_path = os.path.join(config["data"]["data_root_dir"], audio_path)
        if os.path.exists(audio_path):
            audio_check = check_audio_file({"path": audio_path})
            if audio_check["is_audio_ok"]:
                results.append(row.tolist())
            else:
                corrupt_files += 1
        else:
            print(f"File {audio_path} does not exist")
            corrupt_files += 1

    # Convert list of results to new dataframe and overwrite old CSV file
    df_clean = pd.DataFrame(results)
    df_clean.to_csv(csv_file_path, sep=separator, header=False, index=False)

    print(f"Number of corrupt/non-existent files: {corrupt_files}")


def download_blob(bucket_name, source_blob_name, destination_folder):
    """Downloads a blob from the bucket."""

    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)

    storage_client = storage.Client.from_service_account_json(config["gcp_access"])

    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=source_blob_name)  # Get list of files

    for blob in blobs:
        filename = blob.name.replace('/', '_') # replace slashes with underscores
        blob.download_to_filename(destination_folder + filename) # download the file to a destination folder

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_folder
        )
    )
# Example usage:
# download_blob("your-bucket-name", "directory-name/", "/target-directory-path/")


def download_and_extract_drive_file(file_id, destination_folder):
    # Create URL for the file
    url = f"https://drive.google.com/uc?id={file_id}"

    # Download the zip file
    output_path = os.path.join(destination_folder, f"{file_id}.zip")
    gdown.download(url, output_path, quiet=False)

    # Open and extract the zip file
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

    # Remove the zip file after extraction
    os.remove(output_path)

    print(f"Google Drive file {file_id} downloaded and extracted at {destination_folder}.")

# Example usage:
# download_and_extract_drive_file('1h7QgrNfB47Cjq27S9t1VTxE17J-h8k-', '/target-directory-path/')


def create_multispeaker_audio_csv(root_dir, text_csv, train_csv = None, val_test_csv = None):
    # Read CSV into dataframe
    df = pd.read_csv(text_csv)

    # Preprocess CSV file to map index (Key) to Text
    text_dict = df.set_index('Key')['Text'].to_dict()
    # Create an empty list to store the data
    train_data = []
    val_data = []
    speaker_ids = {}
    # Walk through the directory
    for subdir, dirs, files in os.walk(root_dir):
        # Use the name of the subfolder as an index to the CSV
        try:
            key = int(os.path.basename(subdir))
        except ValueError:
            key = None
            
        
        # If the key exists in our CSV
        if key in text_dict:
            for file in files:
                # Check if the file is an audio file
                if file.endswith(".ogg"):
                    # Full path to the file
                    try:
                        sid = speaker_ids[file]
                    except KeyError:
                        speaker_ids[file] = len(speaker_ids)
                        sid = speaker_ids[file]
                    file_path = os.path.join(subdir, file)

                    # Append the path, key and associated text to our data
                    if df.iloc[key]["split"] == "train": 
                      train_data.append([file_path, sid, text_dict[key]])
                    else:
                      val_data.append([file_path, sid, text_dict[key]])
    # Create a dataframe from the data
    train_df_audio = pd.DataFrame(train_data, columns=['Path', 'SID', 'Text'])
    val_df_audio = pd.DataFrame(val_data, columns=['Path', 'SID', 'Text'])

    # Save it to a CSV file
    train_df_audio.to_csv(train_csv, sep='|', index=False, header=None)
    val_df_audio.to_csv(val_test_csv, sep='|', index=False, header=None)


# Usage:
# construct_csv('samples_acholi', 'text.csv', 'output.csv')


