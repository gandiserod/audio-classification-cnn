import os
import random
import numpy as np
import torch
import torchaudio

SEED = 42

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Audio parameters
SAMPLE_RATE = 16000
MAX_LEN = SAMPLE_RATE * 5
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 320

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)

amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
