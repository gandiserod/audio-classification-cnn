import os
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import SAMPLE_RATE, MAX_LEN, mel_transform

def load_audio(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    wav = torch.tensor(wav, dtype=torch.float32)

    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    if wav.shape[0] > MAX_LEN:
        wav = wav[:MAX_LEN]
    else:
        wav = torch.nn.functional.pad(wav, (0, MAX_LEN - wav.shape[0]))

    return wav

def cache_features(df, audio_root, cache_dir, train=True):
    os.makedirs(cache_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Caching features"):
        name = row["slice_file_name"]
        save_path = os.path.join(cache_dir, name.replace(".wav", ".pt"))
        if os.path.exists(save_path):
            continue

        if train:
            fold = int(row["fold"])
            audio_path = os.path.join(audio_root, f"fold{fold}", name)
        else:
            audio_path = os.path.join(audio_root, "test", name)

        wav = load_audio(audio_path)
        mel = mel_transform(wav.unsqueeze(0))
        mel = torch.log(mel + 1e-6)

        delta1 = torchaudio.functional.compute_deltas(mel)
        delta2 = torchaudio.functional.compute_deltas(delta1)

        feat = torch.cat([mel, delta1, delta2], dim=0)
        torch.save(feat, save_path)

class AudioDataset(Dataset):
    def __init__(self, df, cache_dir, train=True):
        self.df = df
        self.cache_dir = cache_dir
        self.train = train
        self.tm = torchaudio.transforms.TimeMasking(30)
        self.fm = torchaudio.transforms.FrequencyMasking(15)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row["slice_file_name"]
        feat = torch.load(os.path.join(self.cache_dir, name.replace(".wav", ".pt")))

        if self.train and torch.rand(1) < 0.5:
            feat = self.tm(feat)
            feat = self.fm(feat)

        if self.train:
            return feat, int(row["classID"])
        else:
            return feat, name
