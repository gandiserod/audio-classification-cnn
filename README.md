# Audio Classification using CNNs (PyTorch)

End-to-end audio classification pipeline using convolutional neural networks trained on log-Mel spectrograms with delta features.

## Highlights
- Log-Mel spectrogram + delta & delta-delta features
- CNN architecture with BatchNorm and Dropout
- Data augmentation via time & frequency masking
- Class-weighted loss for imbalance handling
- Kaggle-style submission generation

## Tech Stack
- Python, PyTorch, Torchaudio
- NumPy, Pandas, scikit-learn

## Usage
```bash
pip install -r requirements.txt
python src/train.py
