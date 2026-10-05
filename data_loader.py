import os

import pandas as pd
import torch
import torchaudio

from torch.utils.data import DataLoader, Dataset


SAMPLE_RATE = 24000
CHUNK_SECONDS = 5
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS


def load_manifest(manifest_path, classes) -> pd.DataFrame:
    manifest_path = os.fspath(manifest_path)
    data = pd.read_csv(manifest_path, keep_default_na=False)

    dataset_dir = os.path.dirname(manifest_path)
    data["_audio_path"] = data["audio_path"].map(lambda path: os.path.join(dataset_dir, path))

    label_to_index = {label: index for index, label in enumerate(classes)}
    data["_label_index"] = (data["label"].map(label_to_index).fillna(-1).astype("int64"))

    return data

class AudioDataset(Dataset):
    def __init__(self, data: pd.DataFrame, split: str):
        self.rows = (data.loc[data["split"] == split].reset_index(drop=True).copy())

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        waveform, sample_rate = torchaudio.load(row["_audio_path"])

        waveform = waveform.mean(dim=0).to(torch.float32).contiguous()

        num_chunks = waveform.numel() // CHUNK_SAMPLES
        waveform = waveform[: num_chunks * CHUNK_SAMPLES]
        chunks = waveform.reshape(num_chunks, CHUNK_SAMPLES)

        label = torch.tensor(int(row["_label_index"]), dtype=torch.long)
        return chunks, label, row["sample_id"]


def get_dataloader(data: pd.DataFrame, split: str, batch_size = 1, num_workers = 0) -> DataLoader:
    dataset = AudioDataset(data, split=split)

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)